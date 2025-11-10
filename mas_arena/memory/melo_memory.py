from collections import defaultdict
from dataclasses import dataclass, replace
import logging
import os
import pickle
import random
import re
from langchain_chroma import Chroma
from chromadb.config import Settings
import copy
import numpy as np
import hdbscan
import networkx as nx
from typing import Iterable
from .utils import write_json, cosine_similarity, load_json, random_divide_list
from .common import MASMessage, RWLock
from .llm import LLMCallable, Message, _get_env_int
from .base import BaseMemory
from langchain.docstore.document import Document
from .prompt import PROMPTS
from .memory_registry import register_memory
import numpy as np
from sklearn.preprocessing import normalize
from typing import List, Optional, Tuple, Dict, Any
logger = logging.getLogger(__name__)

MAX_RULE_THRESHOLD: int = 20

@dataclass
@register_memory('melo_memory')
class MELOMemory(BaseMemory):

    def __post_init__(self):
        
        super().__post_init__()

        self._rw_lock = RWLock()

        self.main_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
            client_settings=Settings(anonymized_telemetry=False)
        )

        self._hop: int = _get_env_int('HOP', 1)
        self._start_insights_threshold: int = _get_env_int('START_INSIGHTS_THRESHOLD', 5)
        self._rounds_per_insights: int = _get_env_int('ROUNDS_PER_INSIGHTS', 5) 
        self._insights_point_num: int = _get_env_int('INSIGHTS_POINT_NUM', 5)

        self.task_layer = TaskLayer(
            working_dir=self.persist_dir,
            namespace='task_layer', 
            task_storage=self.main_memory
        )

        self.insights_layer = InsightsManager(
            working_dir=self.persist_dir, 
            namespace='insights', 
            llm_model=self.llm_model, 
            task_storage=self.main_memory,
            task_layer=self.task_layer
        )

        self._debug: bool = str(os.getenv('MELO_DEBUG', '0')).lower() in ('1', 'true', 'yes', 'y')
        self._debug_max_len: int = int(os.getenv('MELO_DEBUG_MAXLEN', '500'))
        if self._debug:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                debug_log_path = os.path.join(self.persist_dir, 'memory_debug.log')
                if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('memory_debug.log') for h in logger.handlers):
                    fh = logging.FileHandler(debug_log_path, encoding='utf-8')
                    fh.setLevel(logging.DEBUG)
                    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fh.setFormatter(fmt)
                    logger.addHandler(fh)
                logger.setLevel(logging.DEBUG)
                logger.debug('[MELOMemory] Debug logging enabled. persist_dir=%s', self.persist_dir)
            except Exception as e:
                # Do not fail initialization if logging setup has issues
                pass

    def _clip_for_debug(self, text: str) -> str:
        try:
            if text is None:
                return ''
            s = str(text)
            return s if len(s) <= self._debug_max_len else s[: self._debug_max_len] + '...<truncated>'
        except Exception:
            return '<unprintable>'

    def _dbg(self, msg: str, *args):
        if getattr(self, '_debug', False):
            try:
                logger.debug(msg, *args)
            except Exception:
                pass

    async def add_memory(self, mas_message: MASMessage) -> None:

        # Debug: before extraction
        if self._debug:
            self._dbg('[add_memory.start] task_search_keywords=%s label=%s trajectory_len=%s trajectory_preview=%s',
                      mas_message.task_search_keywords,
                      mas_message.label,
                      0 if mas_message.task_trajectory is None else len(mas_message.task_trajectory),
                      self._clip_for_debug(mas_message.task_trajectory))
        
        with self._rw_lock.write_lock():
            mas_message = await self._extract_mas_message(mas_message=mas_message)  
            if self._debug:
                extra_keys = list(mas_message.extra_fields.keys()) if isinstance(mas_message.extra_fields, dict) else []
                self._dbg('[add_memory.after_extract] label=%s keys=%s trajectory_len=%s trajectory_preview=%s',
                        mas_message.label,
                        extra_keys,
                        0 if mas_message.task_trajectory is None else len(mas_message.task_trajectory),
                        self._clip_for_debug(mas_message.task_trajectory))
            
            # add into memory
            await self.task_layer.add_task_node(mas_message.task_search_keywords)
    
            meta_data: dict = MASMessage.to_dict(mas_message)
            memory_doc = Document(
                page_content=mas_message.task_search_keywords,   
                metadata=meta_data
            )
            if mas_message.label == True or mas_message.label == False:
                if self._debug:
                    self._dbg('[add_memory.write] persist_dir=%s page_content=%s meta_keys=%s extra_fields_len=%s',
                            self.persist_dir,
                            self._clip_for_debug(memory_doc.page_content),
                            list(memory_doc.metadata.keys()),
                            len(memory_doc.metadata.get('extra_fields', '') or ''))
                self.main_memory.add_documents([memory_doc])
                if self._debug:
                    try:
                        ids = self.main_memory.get()["ids"]
                        self._dbg('[add_memory.done] total_docs=%s', len(ids) if ids is not None else 'unknown')
                    except Exception:
                        pass
            else:
                raise ValueError('The mas_message must have label!')
            
            # finetune and merge insights
            if self.memory_size >= self._start_insights_threshold and self.memory_size % self._rounds_per_insights == 0:
                await self.insights_layer.finetune_insights(self._insights_point_num)
            if self.memory_size % 20 == 0: 
                await self.insights_layer.merge_insights() 
            
            self._index_done()

    async def _retrieve_memory_raw(
        self, 
        task_search_keywords: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1, 
        insight_windows: int = 10,
        threshold: float = 0.3
    ) -> tuple[list, list, list]:
        def sort_and_filter_by_similarity(docs: list[Document], threshold: float = 0.3) -> list[tuple[Document, float]]:
            result = []
            for doc in docs:
                embedding = self.embedding_func.embed_query(doc.page_content)
                sim = cosine_similarity(origin_embedding, embedding)
                if sim >= threshold:
                    result.append((doc, sim))

            result.sort(key=lambda x: x[1], reverse=True)
            return result

        true_tasks_doc: list[Document] = []
        false_tasks_doc: list[Document] = []
        
        # find related tasks in task layer
        related_point_num: int = max((successful_topk + failed_topk) // 2, 1)
        related_keywords: list[str] = await self.task_layer.retrieve_related_task(task_search_keywords=task_search_keywords, node_num=related_point_num, hop=self._hop)
        for related_keyword in related_keywords:
            doc = self.main_memory.similarity_search(related_keyword, k=1)[0]

            if doc.metadata.get('label') == True:
                true_tasks_doc.append(doc)
            elif doc.metadata.get('label') == False:
                false_tasks_doc.append(doc)
            else:
                raise RuntimeError('The document object\'s metadata should have `label` attribute.')
        
        # If the specified number is not met, fill in the rest using similarity-based augmentation.
        if len(true_tasks_doc) < successful_topk:
            true_tasks_doc = self.main_memory.similarity_search(
                query=task_search_keywords, k=successful_topk, filter={'label': True}
            )
            for doc in true_tasks_doc:
                if doc not in true_tasks_doc:
                    true_tasks_doc.append(doc)
        
        if len(false_tasks_doc) < failed_topk:
            false_tasks_doc = self.main_memory.similarity_search(
                query=task_search_keywords, k=failed_topk, filter={'label': False}
            )
            for doc in false_tasks_doc:
                if doc not in false_tasks_doc:
                    false_tasks_doc.append(doc)

        # order by similarity       
        origin_embedding: list[float] = self.embedding_func.embed_query(task_search_keywords)
        true_tasks_doc_with_score = sort_and_filter_by_similarity(true_tasks_doc, threshold)[:successful_topk]
        false_tasks_doc_with_score = sort_and_filter_by_similarity(false_tasks_doc, threshold)[:failed_topk]

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc, _ in true_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc, _ in false_tasks_doc_with_score:
            meta_data: dict = doc.metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)
        
        # get insights and order by relelvance
        insights_with_score = self.insights_layer.query_insights_with_score(task_search_keywords, top_k=insight_windows)
        insights = [insight for insight, _ in insights_with_score][:insight_windows]

        return true_task_messages, false_task_messages, insights

    async def retrieve_memory(
        self, 
        task_search_keywords: str,  
        task_question: str,
        successful_topk: int = 2, 
        failed_topk: int = 1,
        insight_topk: int = 10,
        threshold: float = 0.3,
        **args
    ) -> tuple[list, list, list]: 
        with self._rw_lock.read_lock():
        # retrieve raw tasks
            successful_task_trajectories: list[MASMessage]
            failed_task_trajectories: list[MASMessage]
            insights: list[str]
            successful_task_trajectories, failed_task_trajectories, insights = await self._retrieve_memory_raw(
                task_search_keywords, 2*successful_topk, 2*failed_topk, 2*insight_topk, threshold)
            
            # retrieve tasks based on task relevance
            importance_score: list[float] = []
            for success_task in successful_task_trajectories:
                prompt: str = PROMPTS["generative_task_user_prompt"].format(
                    trajectory=success_task.task_question + '\n' + success_task.task_trajectory,
                    query_scenario=task_question
                )
                response: str = self.llm_model(messages=[Message('system', PROMPTS["generative_task_system_prompt"]), 
                                                        Message('user', prompt)])
                score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
                importance_score.append(score)
            
            sorted_success_tasks = [task for _, task in sorted(zip(importance_score, successful_task_trajectories), 
                                                            key=lambda x: x[0], reverse=True)]
            top_success_task_trajectories = sorted_success_tasks[:successful_topk]
            
            top_fail_task_trajectories = failed_task_trajectories[:failed_topk]
            
            top_k_insights = insights[:insight_topk]

            return top_success_task_trajectories, top_fail_task_trajectories, top_k_insights


    async def _extract_mas_message(self, mas_message: MASMessage) -> MASMessage:

        mas_message_copy: MASMessage = copy.deepcopy(mas_message)        
        
        trajectory = mas_message_copy.task_trajectory
        
        if mas_message_copy.label == True:
            mas_message_copy.task_trajectory = trajectory

            system_prompt = PROMPTS["extract_true_traj_system_prompt"]
            prompt_template = PROMPTS["extract_true_traj_user_prompt"]

            prompt: str = prompt_template.format(
                task=mas_message_copy.task_search_keywords,
                trajectory=trajectory
            )
            messages: list[Message] = [Message('system', system_prompt), Message('user', prompt)]
            if self._debug:
                self._dbg('[extract_true_traj.request] task=%s prompt_len=%s prompt_preview=%s',
                        mas_message_copy.task_search_keywords,
                        len(prompt),
                        self._clip_for_debug(prompt))
            response: str = self.llm_model(messages, temperature=0.1)
            if self._debug:
                self._dbg('[extract_true_traj.response] key_steps_len=%s key_steps_preview=%s',
                        0 if response is None else len(response),
                        self._clip_for_debug(response))
            
            mas_message_copy.add_extra_field('key_steps', response)


        if mas_message_copy.label == False:
            reason: str = self._detect_mistakes(mas_message_copy)
            mas_message_copy.add_extra_field('fail_reason', reason)
            if self._debug:
                self._dbg('[detect_mistakes.response] fail_reason_len=%s fail_reason_preview=%s',
                          0 if reason is None else len(reason),
                          self._clip_for_debug(reason))
        
        return mas_message_copy
    
    
    async def _detect_mistakes(self, mas_message: MASMessage) -> str:
        user_prompt: str = PROMPTS["detect_mistakes_user_prompt"].format(task=mas_message.task_question, trajectory=mas_message.task_trajectory,ground_truth=mas_message.ground_truth,final_answer=mas_message.final_answer)
        messages: list[Message] = [Message('system',PROMPTS["detect_mistakes_system_prompt"] ), 
                                   Message('user', user_prompt)]
        response: str = self.llm_model(messages)    

        return response

    async def backward(self, reward: bool, insights_from_retrieve: list[str]):
        with self._rw_lock.write_lock():
            for insight in insights_from_retrieve:
                self.insights_layer.backward(insight, reward=-2 if reward == False else 1)

    
    @property
    def memory_size(self):
        num_records = self.main_memory.get()["ids"]
        return len(num_records)
    
@dataclass
class TaskLayer:
    
    working_dir: str
    namespace: str
    task_storage: Chroma
    
    def __post_init__(self):
        self.similarity_threshold = 0.7

        self._graph_pic_save_path: str = os.path.join(self.working_dir, 'graph.png')
        self._node_match_save_path: str = os.path.join(self.working_dir, 'match_nodes.txt')
        self._graph_save_path: str = os.path.join(self.working_dir, f'{self.namespace}_graph.pkl')

        if os.path.exists(self._graph_save_path):
            with open(self._graph_save_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"Graph loaded from {self._graph_save_path}")
        else:
            self.graph = nx.Graph()
            print("New empty graph created")

    async def add_task_node(self, task_search_keywords: str) -> None:
        """Add a task node to the task graph.

        Args:
            task_search_keywords (str): task name
        """
        if task_search_keywords in self.graph:
            return  

        self.graph.add_node(task_search_keywords)

        results: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(
            query=task_search_keywords,
            k=10 
        )
        
        for doc, distance in results:
            similarity = 1 - distance
            if similarity < self.similarity_threshold:
                continue  

            neighbor = doc.page_content

            if neighbor not in self.graph:
                self.graph.add_node(neighbor)

            self.graph.add_edge(task_search_keywords, neighbor, weight=similarity) 
        
        self._index_done()
 
    async def retrieve_related_task(self, task_search_keywords: str, node_num: int, hop: int = 1) -> list[str]:

        tasks: list[tuple[Document, float]] = self.task_storage.similarity_search_with_score(query=task_search_keywords, k=node_num)
        top_nodes = [doc[0].page_content for doc in tasks]

        related_nodes = set(top_nodes)
        for node in top_nodes:
            try:
                neighbours = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
                related_nodes.update(neighbours)
            except Exception as e:
                print(f"Error retrieving related nodes: {e}")
                continue
        return list(related_nodes)
    
    async def cluster_tasks(self) -> None:
        """
        Perform clustering on tasks in the graph using their embeddings and assign cluster IDs.

        This method extracts all nodes from the graph, computes embeddings for each node using the
        task storage's embedding function, and applies the FINCH clustering algorithm with cosine similarity.
        """
        nodes = list(self.graph.nodes)

        embeddings = []
        valid_nodes = []

        for node in nodes:
            embedding = self.task_storage._embedding_function.embed_query(node)  
            if embedding is not None:
                embeddings.append(embedding)
                valid_nodes.append(node)

        X = np.vstack(embeddings)
        X = normalize(X, norm='l2')
        
        clusterer = hdbscan.HDBSCAN(
            metric='euclidean', 
            min_cluster_size=2, 
            allow_single_cluster=True
        )

        try: 
            labels = clusterer.fit_predict(X)
        except Exception as e:   
            print(f"HDBSCAN clustering failed: {e}")
            labels = np.zeros(len(valid_nodes), dtype=int)

        for node, label in zip(valid_nodes, labels):
            self.graph.nodes[node]['cluster_id'] = int(label)
        self._index_done()

    async def _index_done(self) -> None:
        
        with open(self._graph_save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def __iter__(self) -> Iterable[tuple[str, int]]: 
        return ((node, self.graph.nodes[node]['cluster_id']) for node in self.graph.nodes)


@dataclass
class InsightsManager:

    working_dir: str
    namespace: str
    llm_model: LLMCallable
    task_storage: Chroma
    task_layer: TaskLayer
    def __post_init__(self):
        self.persist_file: str = os.path.join(self.working_dir,f'{self.namespace}.json')
        self.insights_memory: list[dict] = load_json(self.persist_file) or []
       
        log_path = os.path.join(self.working_dir, 'insights.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        

    def query_insights_with_score(self, task_search_keywords: str, top_k: int = None) -> list[tuple[str, float]]:

        SUCC_NUM, FAIL_NUM = 4, 2

        related_successful_tasks, related_failed_tasks = self._retrieve_memory(task_search_keywords, successful_topk=SUCC_NUM, failed_topk=FAIL_NUM)
        related_task_search_keywords: list[str] = [task.task_search_keywords for task in related_successful_tasks + related_failed_tasks]
        related_task_search_keywords.append(task_search_keywords)
        insights_score = defaultdict(float)
        for related_task_search_keyword in related_task_search_keywords:
            _, related_insights = self._find_related_insights(task_search_keywords=[related_task_search_keyword])
            for insight in related_insights:
                insights_score[insight.get('rule')] += 1  

        sorted_insights = sorted(insights_score.items(), key=lambda x: x[1], reverse=True) 
        if top_k is not None:
            sorted_insights = sorted_insights[:top_k]
        return sorted_insights
    
    async def merge_insights(self) -> None:

        self.task_layer.cluster_tasks()
        
        label_tasks: dict[int, list[str]] = {}
        for task_search_keywords, label_id in self.task_layer:
            if label_id is None:
                raise RuntimeError('Label id should not be none.')
            if label_id not in label_tasks.keys():
                label_tasks[label_id] = [task_search_keywords]
            else:
                label_tasks[label_id].append(task_search_keywords)
        
        merged_label_rules: dict[int, list[str]] = {}
        for task_type, related_task_search_keywords in label_tasks.items():
            related_ids, related_insights = self._find_related_insights(task_search_keywords=related_task_search_keywords)
            related_rules: list[str] = [insight['rule'] for insight in related_insights]
            merged_rules: list[str] = self._merge_rules(related_rules)
            merged_label_rules[task_type] = merged_rules

            self.logger.info('------- Merge Insights -------')
            self.logger.info(f'Task type: {task_type}')
            origin_rules_str = '\n'.join(related_rules)
            self.logger.info(f"Origin rules: \n{origin_rules_str}")
            merged_rules_str = '\n'.join(merged_rules)
            self.logger.info(f"Merged rules: \n{merged_rules_str}")
            
        self.insights_memory.clear()

        for label, related_rules in merged_label_rules.items():
            related_task_search_keywords = label_tasks.get(label)
            if related_task_search_keywords is None:
                raise RuntimeError('Inconsistency in `label`')
            
            for rule in related_rules:
                insight: dict = {
                    'rule': rule,
                    'score': 2,          
                    'positive_correlation_tasks': list(related_task_search_keywords),
                    'negative_correlation_tasks': list()
                }
                self.insights_memory.append(insight)
        
        self._index_done()

    def _merge_rules(self, rules: list[str]) -> list[str]:
        def parse_numbered_list(text: str) -> list[str]:
            pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
            items = re.findall(pattern, text.strip(), flags=re.DOTALL)
            return [item.strip() for item in items]
        
        merged_rules = []
        batch_size = 10

        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            actual_num: int = len(batch) // 3  

            user_prompt = PROMPTS["merge_rules_user_prompt"].format(
                current_rules='\n'.join(batch),
                limited_number=actual_num//3
            )
            messages = [Message('system', PROMPTS["merge_rules_system_prompt"]),
                        Message('user', user_prompt)]
            raw_merged_rules = self.llm_model(messages)
            merged_rules.extend(parse_numbered_list(raw_merged_rules))

        return merged_rules

    def backward(self, insight: str, reward: float):
        
        for inner_insight in self.insights_memory:
            if insight in inner_insight['rule']:
                inner_insight['score'] += reward

        self.clear_insights()
        self._index_done()

    def clear_insights(self):
        self.insights_memory = [self.insights_memory[i] for i in range(len(self.insights_memory)) 
                        if self.insights_memory[i]['score'] > 0] 

    def _retrieve_memory(
        self,
        query_task: str,   
        successful_topk: int = 1, 
        failed_topk: int = 1
    ) -> tuple[list[MASMessage], list[MASMessage]]:

        true_tasks_doc: list[tuple[Document, float]] = []
        false_tasks_doc: list[tuple[Document, float]] = []

        if successful_topk != 0:
            true_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=successful_topk, filter={'label': True}
            )
        if failed_topk != 0:
            false_tasks_doc = self.task_storage.similarity_search_with_score(
                query=query_task, k=failed_topk, filter={'label': False}
            )
        sorted(true_tasks_doc, key=lambda x: x[1]) 
        sorted(false_tasks_doc, key=lambda x: x[1]) 

        true_task_messages: list[MASMessage] = []
        false_task_messages: list[MASMessage] = []
        for doc in true_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            true_task_messages.append(mas_message)
        
        for doc in false_tasks_doc:
            meta_data: dict = doc[0].metadata
            mas_message: MASMessage = MASMessage.from_dict(meta_data)
            false_task_messages.append(mas_message)

        return true_task_messages, false_task_messages
    
    @property
    def task_size(self):
        num_records = self.task_storage.get()["ids"]
        return len(num_records)
    
    def _find_related_insights(
        self,
        task_search_keywords: list[str],
        threshold: float = 1
    ) -> tuple[list[int], list[dict]]:
        #(rule, score, index)
        rule_set: list[tuple[dict, int, int]] = []  

        for idx, rule in enumerate(self.insights_memory):
            score: int = sum(task_search_keyword in rule.get('positive_correlation_tasks', []) for task_search_keyword in task_search_keywords)
            if score >= threshold:
                rule_set.append((rule, score, idx))

        rule_set.sort(key=lambda x: x[1], reverse=True)

        rule_indices = [item[2] for item in rule_set]
        sorted_rules = [item[0] for item in rule_set]

        return rule_indices, sorted_rules

    async def finetune_insights(self, num_points: int):

        SUCCESS_TASK_NUM, FAIL_TASK_NUM = 3, 1

        all_ids = self.task_storage.get()['ids']
        for _ in range(num_points):  

            random_id = random.choice(all_ids)
            random_entry = self.task_storage.get(ids=[random_id])
            if 'metadatas' in random_entry and random_entry['metadatas']:
                random_metadata = random_entry['metadatas'][0]  
            else:
                raise RuntimeError('Incomplete data.')
            mas_message: MASMessage = MASMessage.from_dict(random_metadata)


            true_trajs, false_trajs = self._retrieve_memory(
                query_task=mas_message.task_search_keywords, successful_topk=SUCCESS_TASK_NUM, failed_topk=FAIL_TASK_NUM
            )
            if mas_message.label == True:
                true_trajs.append(mas_message)
            else:
                false_trajs.append(mas_message)
            all_task_search_keywords: list[str] = [traj.task_search_keywords for traj in true_trajs + false_trajs]

            related_insight_ids, _ = self._find_related_insights(all_task_search_keywords, len(all_task_search_keywords) / 2)
            self._finetune_insights(true_trajs, false_trajs, related_insight_ids)
        
        self.clear_insights()
        self._index_done()

    def _finetune_insights(
        self,
        successful_task_trajectories: list[MASMessage],
        failed_task_trajectories: list[MASMessage],
        insight_ids: list[int]
    ) -> None:

        def map_operations(origin_operations: list[tuple]) -> list[tuple]:
            processed_operations: list[tuple] = []
            for (operation, text) in origin_operations:
                res: list = operation.split(' ')

                if len(res) == 2:
                    if len(insight_ids) == 0:    
                        continue
                    insight_id: int = int(res[1]) - 1
                    if insight_id >= len(insight_ids) or insight_id < 0:
                        continue
                    
                    res[1] = str(insight_ids[insight_id] + 1)   
                    operation: str = ' '.join(res)
                processed_operations.append((operation, text))
            
            return processed_operations

        rule_list: list[dict] = [self.insights_memory[i] for i in insight_ids]

        compare_pairs: list[tuple[MASMessage, MASMessage]] = []
        for id, fail_task in enumerate(failed_task_trajectories):
            if id >= len(successful_task_trajectories):
                break
            success_task = successful_task_trajectories[id]
            compare_pairs.append((success_task, fail_task))
        
        successful_task_chunks: list[list[MASMessage]] = random_divide_list(successful_task_trajectories, 5) 
        
        suffix: str = PROMPTS["finetune_insights_suffix"]['full'] if len(self.insights_memory) > MAX_RULE_THRESHOLD \
                      else PROMPTS["finetune_insights_suffix"]['not_full']


        self.logger.info('--------------- Finetune Insights ---------------')
        for pair in compare_pairs:
            compare_prompts: list[Message] = self._build_comparative_prompts(pair[0], pair[1], rule_list)
            compare_prompts[0] = replace(compare_prompts[0], content=compare_prompts[0].content + suffix)
            response: str = self.llm_model(compare_prompts)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            self._update_rules(
                [pair[0].task_search_keywords, pair[1].task_search_keywords], 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            self.logger.info(compare_prompts[0].role + compare_prompts[0].content + '\n\n' + compare_prompts[1].role + compare_prompts[1].content)
            self.logger.info(response)
            self.logger.info('\n---------------\n')

        for chunk in successful_task_chunks:
            success_prompts: list[Message] = self._build_success_prompts(chunk, rule_list) 
            success_prompts[0] = replace(success_prompts[0], content=success_prompts[0].content + suffix)
            response: str = self.llm_model(success_prompts)
            parsed_operations = self._parse_rules(response)
            processed_operations = map_operations(parsed_operations)
            task_search_keywords: list[str] = [traj.task_search_keywords for traj in chunk]
            self._update_rules(
                task_search_keywords, 
                processed_operations, 
                MAX_RULE_THRESHOLD
            )
            self.logger.info(success_prompts[0].role + success_prompts[0].content + '\n\n' + success_prompts[1].role + success_prompts[1].content)
            self.logger.info(response)
            self.logger.info('\n---------------\n')
        
        self.clear_insights()
        self._index_done()

    def _index_done(self):
        write_json(self.insights_memory, self.persist_file)

    def _build_comparative_prompts(self, true_traj: MASMessage, false_traj: MASMessage, insights: list[dict]) -> list[Message]:
        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        prompt = PROMPTS["critique_compare_rules_user_prompt"].format(   
            task1=true_traj.task_search_keywords,
            task1_trajectory=true_traj.task_trajectory,   
            task2=false_traj.task_search_keywords,
            task2_trajectory=false_traj.task_trajectory,
            fail_reason=false_traj.get_extra_field('fail_reason'),
            existing_rules=rule_text
        )

        return [Message(role='system', content= PROMPTS["critique_compare_rules_system_prompt"]), Message(role='user', content=prompt)] 
    
    def _build_success_prompts(
        self,
        success_trajectories: Iterable[MASMessage],
        insights: list[dict],
    ) -> list[Message]:

        existing_rules: list[str] = [insight['rule'] for insight in insights]
        if len(existing_rules) == 0:
            existing_rules.append('')
        rule_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])

        history: list[str] = [f'task{i}:\n' + task.task_search_keywords + task.get_extra_field('key_steps') for i, task in enumerate(success_trajectories)]
        prompt = PROMPTS["critique_success_rules_user_prompt"].format(
            success_history='\n'.join(history),
            existing_rules=rule_text
        )

        return [Message(role='system', content=PROMPTS["critique_success_rules_system_prompt"]), Message(role='user', content=prompt)]
    
    def _parse_rules(self, llm_text):
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        matches = re.findall(pattern, llm_text)

        res = []
        banned_words = ['ADD', 'AGREE', 'EDIT']
        for operation, text in matches:
            text = text.strip()
            if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):

                if 'ADD' in operation:
                    res.append(('ADD', text))
                else:
                    res.append((operation.strip(), text))
        return(res)
    
    def _update_rules(
        self,
        relative_tasks: list[str],
        operations: list[tuple[str, str]], 
        max_rules_num: int = 10
    ) -> None:

        delete_indices = []
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

            if operation_type == 'ADD':    
                if self._is_existing_rule(operation_rule_text): 
                    delete_indices.append(i)
                    
            elif operation_type == 'EDIT':   
                if self._is_existing_rule(operation_rule_text): 
                    rule_num: int = self._retrieve_rule_index(operation_rule_text)
                    operations[i] = (f'AGREE {rule_num + 1}', operation_rule_text)   

                elif (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
                        
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':  
                if (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):   
                    delete_indices.append(i)
            
            else: 
                delete_indices.append(i)

        operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] 
        

        list_full: bool = len(self.insights_memory) >= max_rules_num  
        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: 
            for i in range(len(operations)):
                operation, operation_rule_text = operations[i]
                operation_type = operation.split(' ')[0]
                if operation_type != op:
                    continue

                if operation_type == 'REMOVE': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    remove_strength = 3 if list_full else 1
                    rule_data['score'] -= remove_strength
                    rule_data['negative_correlation_tasks'] = list(set(rule_data['negative_correlation_tasks'] + relative_tasks))  

                elif operation_type == 'AGREE':
                    rule_index: int = self._retrieve_rule_index(operation_rule_text) 
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'EDIT': 
                    rule_index = int(operation.split(' ')[1]) - 1
                    rule_data: dict = self.insights_memory[rule_index]
                    rule_data['rule'] = operation_rule_text
                    rule_data['score'] += 1
                    rule_data['positive_correlation_tasks'] = list(set(rule_data['positive_correlation_tasks'] + relative_tasks))

                elif operation_type == 'ADD': 
                    meta_data: dict = {
                        'rule': operation_rule_text,
                        'score': 2,         
                        'positive_correlation_tasks': list(relative_tasks),
                        'negative_correlation_tasks': list()
                    }
                    self.insights_memory.append(meta_data)

    def _is_existing_rule(self, operation_rule_text: str) -> bool:

        for insight in self.insights_memory:
            if insight['rule'] in operation_rule_text:
                return True
        return False
    
    def _retrieve_rule_index(self, operation_rule_text: str) -> int:

        for idx, insight in enumerate(self.insights_memory):
            if insight['rule'] in operation_rule_text:
                return idx
        return -1
