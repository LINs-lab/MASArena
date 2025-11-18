# Failure Attribution Module

This module provides automated failure attribution capabilities for analyzing multi-agent system responses and identifying failure causes. The module has been migrated and adapted from the [`Automated_FA`](https://github.com/mingyin1/Agents_Failure_Attribution).

## Overview

The failure attribution module analyzes agent conversation histories to identify:
- Which agent made an error
- At which step the error occurred
- What type of error it was
- The specific reason for the failure


## Usage

The module can be run using the `inference.py` script with various analysis methods:

```bash
python inference.py --method all_at_once --model gpt-4o --directory_path ../results/agent_responses
```

### Analysis Methods

1. **All-at-once Analysis**: Analyzes the entire conversation history at once
```bash
python inference.py --method all_at_once --model gpt-4o
```

2. **Step-by-step Analysis**: Analyzes the conversation incrementally, step by step
```bash
python inference.py --method step_by_step --model gpt-4o
```

3. **Binary Search Analysis**: Uses binary search to efficiently locate errors
```bash
python inference.py --method binary_search --model gpt-4o
```


### Output

The analysis results are saved to the `outputs/` directory with filenames in the format:
`{method}_{model}_agent_responses.txt`

Example output format:
```
Error Agent: agent_2
Error Step: 3
Error Type: Calculation Error
Reason: The agent made an arithmetic error in the calculation step.
```



## Data Format (agent_responses)
- Uses `responses` field for agent interactions
- No ground truth labels (unsupervised analysis)
- Includes `problem_id` and `agent_system` metadata
- Each response contains `agent_id`, `content`, and `timestamp`

```json
{
    "problem_id": "problem_1",
    "agent_system": "multi_agent",
    "run_id": "run_123",
    "timestamp": "2024-06-24T16:11:44",
    "responses": [
        {
            "timestamp": "2024-06-24T16:11:44.123",
            "problem_id": "problem_1",
            "message_index": 0,
            "agent_id": "agent_1",
            "content": "Agent response content here...",
            "role": "assistant",
            "message_type": "response",
            "usage_metadata": {}
        }
    ]
}
```

## Evaluation

To evaluate the accuracy of failure attribution predictions:

```bash
python evaluate.py --data_path /path/to/annotated/data --evaluation_file outputs/all_at_once_gpt-4o_agent_responses.txt
```

The evaluation script compares predictions against ground truth annotations and reports:
- Agent identification accuracy
- Error step identification accuracy



## Troubleshooting

### Performance Tips

- Use `all_at_once` for comprehensive analysis
- Use `binary_search` for efficient error localization in long conversations
- Use `step_by_step` for detailed incremental analysis
- For local models, ensure adequate GPU memory or use CPU inference
