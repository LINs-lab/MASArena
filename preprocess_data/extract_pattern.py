import json
from process_pipeline import _parse_analysis_to_dict


def main():
    pattern_file = 'preprocess_data/question_pattern_completion.jsonl'
    patterns = []
    with open(pattern_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            patterns.append(data)
    print(patterns)
    count:int = 0
    for pattern in patterns:
        if 'pattern_analysis' in pattern:
            if 'pattern'  not in pattern['pattern_analysis']:
                analysis_text = pattern['raw_analysis_text']
                question_id = pattern['question_id']
                question_text = pattern['question']
                analysis_dict = _parse_analysis_to_dict(analysis_text)
                if 'pattern' not in analysis_dict.keys():
                    count += 1
                    print(question_id)
                # return {
                #     "question_id": question_id,
                #     "question": question_text,
                #     "pattern_analysis": analysis_dict if analysis_dict else analysis_text,
                #     "raw_analysis_text": analysis_text 
                # }
    print(f"Total patterns with empty pattern: {count}")

if __name__ == "__main__":
    main()
