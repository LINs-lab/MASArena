import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

def main():
    questions_file = 'preprocess_data/gaia_questions.jsonl'
    validate_file = 'data/gaia_validate.jsonl'
    test_file = 'data/gaia_test.jsonl'
    embeddings_file = 'preprocess_data/gaia_questions_embeddings.npy'
    similarity_file = 'preprocess_data/gaia_questions_similarity.npy'
    
    questions = []
    with open(validate_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['Question'])
    
    with open(test_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['Question'])
    
    with open(questions_file, 'w', encoding='utf-8') as f:
        for item in questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    '''
    questions_file = 'preprocess_data/knowledge_base.json'
    with open('preprocess_data/knowledge_base.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    questions = [item['question'] for item in data]
    print(f"question size:{len(questions)}")
    '''

    '''
    plans = []
    for item in data:
        plans.append(item['agent_planning']+ "\n\n" + item['search_agent_planning'])
    print(f"plan size:{len(plans)}")
  
    embeddings_file = 'preprocess_data/knowledge_base_plan_embeddings.npy'
    similarity_file = 'preprocess_data/knowledge_base_plan_similarity.npy'
    '''
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the questions into vectors
    embeddings = model.encode(questions, convert_to_tensor=True)

    # Calculate cosine similarity between all pairs of questions
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Save the embeddings and similarity matrix
    np.save(embeddings_file, embeddings.cpu().numpy())
    np.save(similarity_file, cosine_scores.cpu().numpy())

    print(f"Embeddings saved to {embeddings_file}")
    print(f"Similarity matrix saved to {similarity_file}")
    
if __name__ == "__main__":
    main()