# -*- coding: utf-8 -*-
import os
import json
import datetime
import random
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.schema import Document
import openai

from .base import BaseMemory
from .common import MASMessage, RWLock
from .llm import LLMCallable, Message
from .memory_registry import register_memory
from .prompt import PROMPTS


VECTOR_SEARCH_TOP_K = 3
CHUNK_SIZE = 256
EMBEDDING_MODEL_CN = 'all-MiniLM-L6-v2'
EMBEDDING_DEVICE = 'cpu'
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese',
    'paraphrase-multilingual-MiniLM-L12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    'm3e-base': 'moka-ai/m3e-base',
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
}


def forgetting_curve(t, S):
    return math.exp(-t / (5 * S))

class MemoryForgetterLoader(UnstructuredFileLoader):
    def __init__(self, filepath, language, mode="elements"):
        self.filepath = filepath
        self.language = language
        self.memory_bank = {}

    def _get_date_difference(self, date1: str, date2: str) -> int:
        date_format = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(date1, date_format)
        d2 = datetime.datetime.strptime(date2, date_format)
        return abs((d2 - d1).days)

    def update_memory_when_searched(self, recalled_memos, user, cur_date):
        for recalled in recalled_memos:
            recalled_id = recalled.metadata['memory_id']
            recalled_date = recalled_id.split('_')[1]
            if user not in self.memory_bank or 'history' not in self.memory_bank[user] or recalled_date not in self.memory_bank[user]['history']:
                continue
            for i, memory in enumerate(self.memory_bank[user]['history'][recalled_date]):
                if memory.get('memory_id') == recalled_id:
                    self.memory_bank[user]['history'][recalled_date][i]['memory_strength'] += 1
                    self.memory_bank[user]['history'][recalled_date][i]['last_recall_date'] = cur_date
                    break

    def write_memories(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.memory_bank, f, ensure_ascii=False, indent=4)

    def load_memories(self):
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            self.memory_bank = {}
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)

    def apply_forgetting_and_get_docs(self, user_name, now_date):
        self.load_memories()
        docs = []
        if user_name not in self.memory_bank:
            return docs

        user_memory = self.memory_bank.get(user_name, {})
        if 'history' not in user_memory:
            return docs

        dates_to_delete = []
        for date, content in user_memory.get('history', {}).items():
            forget_ids = []
            for i, dialog in enumerate(content):
                memory_strength = dialog.get('memory_strength', 1)
                last_recall_date = dialog.get('last_recall_date', date)
                memory_id = dialog.get('memory_id', f'{user_name}_{date}_{i}')

                self.memory_bank[user_name]['history'][date][i].setdefault('memory_strength', memory_strength)
                self.memory_bank[user_name]['history'][date][i].setdefault('last_recall_date', last_recall_date)
                self.memory_bank[user_name]['history'][date][i].setdefault('memory_id', memory_id)

                days_diff = self._get_date_difference(last_recall_date, now_date)
                retention_probability = forgetting_curve(days_diff, memory_strength)

                if random.random() > retention_probability:
                    forget_ids.append(i)
                else:
                    tmp_str = f"User: {dialog['query']}; AI: {dialog['response']}"
                    metadata = {'memory_strength': memory_strength, 'memory_id': memory_id, 'last_recall_date': last_recall_date, "source": date}
                    docs.append(Document(page_content=tmp_str, metadata=metadata))

            if len(forget_ids) > 0:
                for idd in sorted(forget_ids, reverse=True):
                    self.memory_bank[user_name]['history'][date].pop(idd)

            if not self.memory_bank[user_name]['history'][date]:
                dates_to_delete.append(date)

        for date in dates_to_delete:
            self.memory_bank[user_name]['history'].pop(date, None)
            self.memory_bank[user_name].get('summary', {}).pop(date, None)

        self.write_memories()
        return docs

@dataclass
@register_memory('memory_bank')
class MemoryBank(BaseMemory):
    language: str = 'cn'
    embedding_model_name: str = EMBEDDING_MODEL_CN
    top_k: int = VECTOR_SEARCH_TOP_K

    def __post_init__(self):
        super().__post_init__()
        self._rw_lock = RWLock()
        self.memory_file_path = os.path.join(self.persist_dir, f'{self.namespace}_memory.json')
        self.vs_path = os.path.join(self.persist_dir, f'{self.namespace}_faiss_index')
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[self.embedding_model_name],
            model_kwargs={'device': EMBEDDING_DEVICE}
        )
        
        self.memory_loader = MemoryForgetterLoader(self.memory_file_path, self.language)
        self.vector_store = self._init_memory_vector_store()
        self.summarize_interval = 10
        self.memory_add_count = 0

    def _init_memory_vector_store(self) -> Optional[FAISS]:
        cur_date = datetime.date.today().strftime("%Y-%m-%d")
        docs = self.memory_loader.apply_forgetting_and_get_docs(self.namespace, cur_date)

        if not docs:
            if os.path.exists(self.vs_path):
                return FAISS.load_local(self.vs_path, self.embeddings, allow_dangerous_deserialization=True)
            return None

        if os.path.exists(self.vs_path):
            vector_store = FAISS.load_local(self.vs_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(docs)
        else:
            vector_store = FAISS.from_documents(docs, self.embeddings)
        
        vector_store.save_local(self.vs_path)
        return vector_store

    async def add_memory(self, mas_message: MASMessage):
        with self._rw_lock.write_lock():
            self.memory_loader.load_memories()
            memory_bank = self.memory_loader.memory_bank
            user_memory = memory_bank.setdefault(self.namespace, {'history': {}, 'summary': {}})
            
            cur_date = datetime.date.today().strftime("%Y-%m-%d")
            date_history = user_memory['history'].setdefault(cur_date, [])
            
            new_dialog = {
                'query': mas_message.task_question,
                'response': mas_message.final_answer,
                'memory_strength': 1,
                'last_recall_date': cur_date,
                'memory_id': f'{self.namespace}_{cur_date}_{len(date_history)}'
            }
            date_history.append(new_dialog)
            self.memory_loader.write_memories()
            
            doc = Document(
                page_content=f"User: {new_dialog['query']}; AI: {new_dialog['response']}",
                metadata={'memory_strength': new_dialog['memory_strength'], 'memory_id': new_dialog['memory_id'], 'last_recall_date': new_dialog['last_recall_date'], "source": cur_date}
            )

            if self.vector_store:
                self.vector_store.add_documents([doc])
            else:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            
            self.vector_store.save_local(self.vs_path)

            self.memory_add_count += 1
            if self.memory_add_count % self.summarize_interval == 0:
                await self._summarize_memory_for_date(cur_date)

    async def retrieve_memory(self, task_search_keywords: str, **kwargs) -> tuple[list, list, list]:
        with self._rw_lock.read_lock():
            if not self.vector_store:
                return [], [], []

            cur_date = datetime.date.today().strftime("%Y-%m-%d")
            related_docs_with_score = self.vector_store.similarity_search_with_score(task_search_keywords, k=self.top_k)
            
            related_docs = [doc for doc, score in related_docs_with_score]
            
            self.memory_loader.update_memory_when_searched(related_docs, self.namespace, cur_date)
            self.memory_loader.write_memories()

            # The base class expects a tuple of three lists.
            # This memory type mainly provides conversational history, not structured "insights" or "failed tasks".
            # We will return the retrieved documents as the first element.
            return related_docs, [], []

    async def _summarize_memory_for_date(self, date: str):
        self.memory_loader.load_memories()
        user_memory = self.memory_loader.memory_bank.get(self.namespace, {})
        history_for_date = user_memory.get('history', {}).get(date, [])

        if not history_for_date:
            return

        prompt_str = PROMPTS.get("summarize_memory_prompt_cn" if self.language == 'cn' else "summarize_memory_prompt_en", 
                                "Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. Dialogue content:\n{dialogues}")
        
        dialogues = ""
        for dialog in history_for_date:
            dialogues += f"\nUser: {dialog['query']}\nAI: {dialog['response']}"

        prompt = prompt_str.format(dialogues=dialogues)
        
        summary = self.llm_model([Message('user', prompt)])
        
        user_memory.setdefault('summary', {})[date] = summary
        self.memory_loader.write_memories()
