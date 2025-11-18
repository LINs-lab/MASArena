from mas_arena.memory.common import MASMessage
import json
import os
import yaml

def load_prompt(prompt_name):
    prompt_path = os.path.join(os.path.dirname(__file__), f"voyager_prompt/{prompt_name}.yaml")
    with open(prompt_path, 'r') as f:
        return f.read()

def load_control_primitives():
    # This is a placeholder implementation.
    # It should return a list of primitive skill strings.
    return []

def dump_text(text, file_path):
    with open(file_path, 'w') as f:
        f.write(text)

class ChestMemory:
    def __init__(self, ckpt_dir="ckpt", resume=False):
        self.ckpt_dir = ckpt_dir
        os.makedirs(f"{ckpt_dir}/action", exist_ok=True)
        if resume:
            print(f"Loading Chest Memory from {ckpt_dir}/action")
            chest_memory_path = f"{ckpt_dir}/action/chest_memory.json"
            if os.path.exists(chest_memory_path):
                self.chest_memory = load_json(chest_memory_path)
            else:
                self.chest_memory = {}
        else:
            self.chest_memory = {}

    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(
                        f"Chest Memory removing chest {position}: {chest}"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"Chest Memory saving chest {position}: {chest}")
                    self.chest_memory[position] = chest
        write_json(self.chest_memory, f"{self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

from langchain.vectorstores import Chroma

from .base import BaseMemory
from .memory_registry import register_memory
from mas_arena.memory.llm import LLMCallable, Message
from mas_arena.memory.utils import EmbeddingFunc, load_json, write_json


class SkillManager:
    def __init__(
        self,
        retrieval_top_k=5,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        llm_model: LLMCallable = None,
        embedding_func: EmbeddingFunc = None,
    ):
        self.llm = llm_model
        os.makedirs(f"{ckpt_dir}/skill/code", exist_ok=True)
        os.makedirs(f"{ckpt_dir}/skill/description", exist_ok=True)
        os.makedirs(f"{ckpt_dir}/skill/vectordb", exist_ok=True)
        # programs for env execution
        self.control_primitives = load_control_primitives()
        if resume:
            print(f"Loading Skill Manager from {ckpt_dir}/skill")
            skills_path = f"{ckpt_dir}/skill/skills.json"
            if os.path.exists(skills_path):
                self.skills = load_json(skills_path)
            else:
                self.skills = {}
        else:
            self.skills = {}
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=embedding_func,
            persist_directory=f"{ckpt_dir}/skill/vectordb",
        )
        assert self.vectordb._collection.count() == len(self.skills), (
            f"Skill Manager's vectordb is not synced with skills.json.\n"
            f"There are {self.vectordb._collection.count()} skills in vectordb but {len(self.skills)} skills in skills.json.\n"
            f"Did you set resume=False when initializing the manager?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    @property
    def programs(self):
        programs = ""
        for skill_name, entry in self.skills.items():
            programs += f"{entry['code']}\n\n"
        for primitives in self.control_primitives:
            programs += f"{primitives}\n\n"
        return programs

    def add_new_skill(self, info):
        if info["task"].startswith("Deposit useless items into the chest at"):
            # No need to reuse the deposit skill
            return
        program_name = info["program_name"]
        program_code = info["program_code"]
        skill_description = self.generate_skill_description(program_name, program_code)
        print(
            f"Skill Manager generated description for {program_name}:\n{skill_description}"
        )
        if program_name in self.skills:
            print(f"Skill {program_name} already exists. Rewriting!")
            self.vectordb._collection.delete(ids=[program_name])
            i = 2
            while f"{program_name}V{i}.js" in os.listdir(f"{self.ckpt_dir}/skill/code"):
                i += 1
            dumped_program_name = f"{program_name}V{i}"
        else:
            dumped_program_name = program_name
        self.vectordb.add_texts(
            texts=[skill_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )
        self.skills[program_name] = {
            "code": program_code,
            "description": skill_description,
        }
        assert self.vectordb._collection.count() == len(
            self.skills
        ), "vectordb is not synced with skills.json"
        dump_text(
            program_code, f"{self.ckpt_dir}/skill/code/{dumped_program_name}.js"
        )
        dump_text(
            skill_description,
            f"{self.ckpt_dir}/skill/description/{dumped_program_name}.txt",
        )
        write_json(self.skills, f"{self.ckpt_dir}/skill/skills.json")
        self.vectordb.persist()

    def generate_skill_description(self, program_name, program_code):
        messages = [
            Message(role="system", content=load_prompt("skill")),
            Message(
                role="user",
                content=program_code
                + "\n\n"
                + f"The main function is `{program_name}`."
            ),
        ]
        skill_description = f"    // { self.llm(messages)}"
        return f"async function {program_name}(bot) {{\n{skill_description}\n}}"

    def retrieve_skills(self, query):
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        print(f"Skill Manager retrieving for {k} skills")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        print(
            f"Skill Manager retrieved skills: "
            f"{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}"
        )
        skills = []
        for doc, _ in docs_and_scores:
            skills.append(self.skills[doc.metadata["name"]]["code"])
        return skills


@register_memory("voyager")
class VoyagerMemory(BaseMemory):

    def __init__(self,
                 llm_model: LLMCallable,
                 embedding_func: EmbeddingFunc,
                 persist_dir: str,
                 **kwargs):
        super().__init__(llm_model=llm_model, embedding_func=embedding_func, persist_dir=persist_dir, **kwargs)
        self.skill_manager = SkillManager(
            llm_model=self.llm_model,
            embedding_func=self.embedding_func,
            ckpt_dir=self.persist_dir,
            resume=True
        )
        self.chest_memory = ChestMemory(ckpt_dir=self.persist_dir, resume=True)

    async def add_memory(self, mas_message: "MASMessage"):
        if mas_message.label:
            program_name = mas_message.get_extra_field("program_name")
            program_code = mas_message.get_extra_field("program_code")

            if program_name and program_code:
                info = {
                    "task": mas_message.task_question,
                    "program_name": program_name,
                    "program_code": program_code,
                }
                self.skill_manager.add_new_skill(info)

    async def retrieve_memory(self, query: str, **kargs) -> tuple[list, list, list]:
        # Retrieve skills based on the query.
        # The other two lists are for compatibility with BaseMemory, returning empty.
        skills = self.skill_manager.retrieve_skills(query)
        return skills, [], []

    async def update_memory(self, query: str, **kargs) -> None:
        # Voyager memory doesn't have a direct concept of "updating" memory
        # in the same way as other memory systems might. Skills are added, not updated.
        pass
