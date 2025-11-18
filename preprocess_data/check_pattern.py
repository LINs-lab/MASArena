import json

def main():
    pattern_file = 'preprocess_data/question_pattern_2.jsonl'
    patterns = []
    with open(pattern_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            patterns.append(data)
    count:int = 0
    for pattern in patterns:
        if 'pattern_analysis' in pattern:
            if 'pattern' in pattern['pattern_analysis']:
                count += 1

    print(f"Total patterns with pattern: {count}")
if __name__ == "__main__":
    main()
