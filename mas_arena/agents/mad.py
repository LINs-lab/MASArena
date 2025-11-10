import json
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

load_dotenv(override=True)

@dataclass
class DebateAgent:
    """Represents a debate participant"""
    agent_id: str
    name: str
    model_name: str
    temperature: float
    memory_lst: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.memory_lst is None:
            self.memory_lst = []
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            request_timeout=60,
            max_retries=2
        )

    def set_meta_prompt(self, meta_prompt: str):
        """Set meta prompt"""
        self.memory_lst.append({"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        """Add new event to memory"""
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        """Add generated response to memory"""
        self.memory_lst.append({"role": "assistant", "content": memory})

    async def ask(self):
        """Query and get response"""
        from langchain_core.messages import AIMessage
        
        messages = []
        for msg in self.memory_lst:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        response = await self.llm.ainvoke(messages)
        response.name = self.name
        response.id = self.agent_id
        return response

class MADAgent(AgentSystem):
    """Multi-Agent Debate system"""
    
    def __init__(self, name: str = "mad", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.num_players = self.config.get("num_players", 3)
        self.max_round = self.config.get("max_round", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0)
        
        # Debate configuration
        self.debate_config = {
            
    "debate_topic": "",
    "base_answer": "",
    "debate_answer": "",
    "player_meta_prompt": "You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer.\nThe debate topic is stated as follows:\n##debate_topic##",
    "moderator_meta_prompt": "You are a moderator overseeing a debate on the topic: \"##debate_topic##\". Your role is to evaluate arguments and determine the correct answer. **IMPORTANT: You must output your decision in a strict JSON format. The final answer within the JSON must EXACTLY follow the format: {format_prompt}. All backslashes in the answer must be escaped (e.g., use `\\\\` for a single backslash).**",
    "affirmative_prompt": "##debate_topic##",
    "negative_prompt": "##aff_ans##\n\nYou disagree with my answer. Provide your answer and reasons.",
    "moderator_prompt": "Now the ##round## round of debate for both sides has ended.\n\nAffirmative side arguing:\n##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nYou, as the moderator, will evaluate both sides' answers. If a clear preference emerges, summarize your reasons, declare the supported side, and provide the final correct answer. If not, the debate continues. **Please output your decision strictly in JSON format as follows, with no additional text: {{\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"Your reason here.\", \"debate_answer\": \"The final answer here, escaping backslashes.\"}}**",
    "judge_prompt_last1": "Affirmative side arguing: ##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nNow, what answer candidates do we have? Present them without reasons.",
    "judge_prompt_last2": "Therefore, ##debate_topic##\nPlease summarize your reasons and give the final answer that you think is correct. **IMPORTANT: You must output your decision in a strict JSON format as follows, with no additional text: {{\"Whether there is a preference\": \"Yes\", \"Supported Side\": \"Affirmative or Negative based on your judgement\", \"Reason\": \"Your reason here.\", \"debate_answer\": \"The final answer here, escaping backslashes.\"}}**",
    "debate_prompt": "##oppo_ans##\n\nDo you agree with my perspective? Provide your reasons and your answer. **IMPORTANT: The debate_answer must EXACTLY follow the format: {format_prompt}.** The latex format requires **two backslashes to be output**"
    }
        
        # Initialize components
        agent_components = self._create_agents()
        self.players = [w for w in agent_components["workers"] if isinstance(w, DebateAgent)]

    def _create_agents(self) -> Dict[str, List]:
        """Create debate participants and result extractor"""
        name_list = ["Affirmative side", "Negative side", "Moderator"]
        
        players = []
        for i, name in enumerate(name_list):
            agent = DebateAgent(
                agent_id=f"agent_{i+1}",
                name=name,
                model_name=self.model_name,
                temperature=self.temperature
            )
            players.append(agent)

        return {
            "workers": players
        }

    def init_prompt(self, debate_topic: str):
        """Initialize and replace placeholders in prompt templates"""
        config = self.debate_config.copy()
        for key in config:
            if isinstance(config[key], str):
                config[key] = config[key].replace("##debate_topic##", debate_topic)
        return config

    def round_dct(self, num: int) -> str:
        """Convert number to ordinal word"""
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct.get(num, str(num))

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run debate process"""
        
        problem_text = problem["problem"]
        
        # Store all LLM responses
        all_messages = []
        
        # Use format_prompt as debate topic
        debate_topic = f"{problem_text}\n\n{self.format_prompt}" if self.format_prompt else problem_text
        
        # Initialize prompts
        config = self.init_prompt(debate_topic)
        
        # Get participants
        affirmative = self.players[0]
        negative = self.players[1] 
        moderator = self.players[2]
        
        # Clear memory for each agent to prevent context overflow
        affirmative.memory_lst.clear()
        negative.memory_lst.clear()
        moderator.memory_lst.clear()
        
        # Set meta prompts
        affirmative.set_meta_prompt(config['player_meta_prompt'])
        negative.set_meta_prompt(config['player_meta_prompt'])
        moderator.set_meta_prompt(config['moderator_meta_prompt'])
        
        # First round debate
        affirmative.add_event(config['affirmative_prompt'])
        aff_response = await affirmative.ask()
        affirmative.add_memory(aff_response.content)
        all_messages.append(aff_response)
        aff_ans = aff_response.content
        
        negative.add_event(config['negative_prompt'].replace('##aff_ans##', aff_ans))
        neg_response = await negative.ask()
        negative.add_memory(neg_response.content)
        all_messages.append(neg_response)
        neg_ans = neg_response.content
        
        moderator.add_event(config['moderator_prompt'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans).replace('##round##', 'first'))
        mod_response = await moderator.ask()
        moderator.add_memory(mod_response.content)
        all_messages.append(mod_response)
        
        try:
            mod_ans = json.loads(mod_response.content)
        except:
            mod_ans = {"debate_answer": "", "Whether there is a preference": "No"}
        
        # Multi-round debate
        for round_num in range(2, self.max_round + 1):
            if mod_ans.get("debate_answer", "") != "":
                break
                
            affirmative.add_event(config['debate_prompt'].replace('##oppo_ans##', neg_ans))
            aff_response = await affirmative.ask()
            affirmative.add_memory(aff_response.content)
            all_messages.append(aff_response)
            aff_ans = aff_response.content
            
            negative.add_event(config['debate_prompt'].replace('##oppo_ans##', aff_ans))
            neg_response = await negative.ask()
            negative.add_memory(neg_response.content)
            all_messages.append(neg_response)
            neg_ans = neg_response.content
            
            moderator.add_event(config['moderator_prompt'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans).replace('##round##', self.round_dct(round_num)))
            mod_response = await moderator.ask()
            moderator.add_memory(mod_response.content)
            all_messages.append(mod_response)
            
            try:
                mod_ans = json.loads(mod_response.content)
            except:
                mod_ans = {"debate_answer": "", "Whether there is a preference": "No"}
        
        # If still no consensus, use judge
        final_answer = mod_ans.get("debate_answer", "")
        if not final_answer:
            judge = DebateAgent(
                agent_id="judge",
                name="Judge",
                model_name=self.model_name,
                temperature=self.temperature
            )
            judge.set_meta_prompt(config['moderator_meta_prompt'])
            
            # Get final answer candidates
            judge.add_event(config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            judge_response1 = await judge.ask()
            judge.add_memory(judge_response1.content)
            all_messages.append(judge_response1)
            
            # Select final answer
            judge.add_event(config['judge_prompt_last2'])
            judge_response2 = await judge.ask()
            judge.add_memory(judge_response2.content)
            all_messages.append(judge_response2)
            
            final_answer = judge_response2.content
        
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

# Register agent system
AgentSystemRegistry.register(
    "mad",
    MADAgent,
    num_players=3,
    max_round=3,
    temperature=0
) 