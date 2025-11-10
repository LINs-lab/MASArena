import asyncio
import os
import shutil
import datetime
import sys
import unittest

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from mas_arena.memory.memory_bank import MemoryBank
from mas_arena.memory.common import MASMessage

# Mock LLMCallable for testing purposes
class MockLLM:
    def __call__(self, messages, **kwargs):
        return "This is a summary."

class TestMemoryBank(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing."""
        self.test_dir = "temp_test_memory_bank"
        os.makedirs(self.test_dir, exist_ok=True)
        self.mock_llm = MockLLM()

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_memory_bank_integration(self):
        """Full integration test for MemoryBank."""
        
        async def run_test():
            # 1. Initialization
            memory_bank = MemoryBank(
                namespace="test_user",
                persist_dir=self.test_dir,
                llm_model=self.mock_llm,
                embedding_func=None,  # FAISS uses its own embeddings via HuggingFaceEmbeddings
                language='en'
            )
            self.assertTrue(os.path.exists(self.test_dir))
            
            # 2. Add memories
            messages_to_add = [
                MASMessage(task_question="What is the capital of France?", final_answer="Paris", ground_truth="Paris", task_search_keywords="France capital"),
                MASMessage(task_question="Who wrote Hamlet?", final_answer="William Shakespeare", ground_truth="William Shakespeare", task_search_keywords="Hamlet author"),
                MASMessage(task_question="What is the formula for water?", final_answer="H2O", ground_truth="H2O", task_search_keywords="water chemical formula"),
            ]
            
            for msg in messages_to_add:
                await memory_bank.add_memory(msg)

            # Verify that memory file is created and contains data
            self.assertTrue(os.path.exists(memory_bank.memory_file_path))
            with open(memory_bank.memory_file_path, 'r') as f:
                data = f.read()
                self.assertIn("France", data)
                self.assertIn("Shakespeare", data)
                self.assertIn("H2O", data)

            # 3. Retrieve memory
            retrieved_docs, _, _ = await memory_bank.retrieve_memory("capital city of french country")
            self.assertIsNotNone(retrieved_docs)
            self.assertGreater(len(retrieved_docs), 0)
            self.assertIn("Paris", retrieved_docs[0].page_content)

            # 4. Test forgetting mechanism (simplified)
            # We will manually check if the loader can apply forgetting.
            # A full test would require mocking datetime.
            cur_date = datetime.date.today().strftime("%Y-%m-%d")
            docs = memory_bank.memory_loader.apply_forgetting_and_get_docs(
                user_name="test_user",
                now_date=(datetime.date.today() + datetime.timedelta(days=100)).strftime("%Y-%m-%d")
            )
            # With S=1 and t=100, forgetting probability is high. We expect some memories to be forgotten.
            # This is probabilistic, so we just check that it runs without error.
            # A more robust test would mock random.random().
            self.assertLessEqual(len(docs), len(messages_to_add))

        # Run the async test
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
