
PROMPTS = {}
PROMPTS["detect_mistakes_system_prompt"] = """
You are an expert error analyst for hierarchical research agents. Your task is to diagnose why a research agent's final answer is incorrect by comparing it with the ground truth. Focus on identifying logical, factual, or classification errors in the agent's reasoning or data processing.

📌 RULES:
1. You MUST be given the ground truth to perform error analysis.
2. Analyze the agent's full trajectory: look for misclassification (e.g., counting a compilation as a studio album), data truncation, source misinterpretation, or faulty filtering logic.
3. Ignore token/step count — focus on semantic/logical errors.
4. Be concise: output a single-sentence diagnosis.
"""

PROMPTS["detect_mistakes_user_prompt"] = """
Identify the root cause of the error by comparing the agent's final answer with the ground truth.

## Example 1:
### Task
How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.

### Trajectory
The Manager Agent delegates the task to the Search Agent. The Search Agent successfully extracts a discography list from Discogs but includes "Acústico (2003)" — a live album — in the studio album count. The Manager Agent blindly trusts the list and outputs 4.

### Ground Truth
3

### Agent's Final Answer
4

### Error Analysis
Misclassification error: Agent incorrectly counted "Acústico (2003)" — a live album — as a studio album.

---

## YOUR TURN:
### Task
{task}

### Trajectory
{trajectory}

### Ground Truth
{ground_truth}

### Agent's Final Answer
{final_answer}

### Error Analysis
"""

PROMPTS["extract_true_traj_system_prompt"]="""
You are an expert analyst of hierarchical AI agent systems. Your task is to distill a complex, multi-step research trajectory into its essential strategy and critical decision points. Focus on the agent's REASONING, STRATEGY, and KEY OUTCOMES, not just the raw sequence of tool calls.
"""

PROMPTS["extract_true_traj_user_prompt"]="""
📌 RULES:
1. DO NOT add any step not present in the original trajectory.
2. DO NOT filter out "failed" tool calls (e.g., "No page found") — these are critical for understanding the agent's problem-solving and fallback strategies.
3. Clearly separate the roles and outputs of the MANAGER AGENT and the SEARCH AGENT.
4. Focus on summarizing the RESEARCH STRATEGY and KEY DECISIONS, not just listing tool calls. Explain the "why" behind actions (e.g., “Fallback after Wikipedia search failed”).
5. If there is a discrepancy between the Search Agent’s detailed output and external knowledge or the Manager’s final answer, HIGHLIGHT it as a “Potential Inconsistency”.
6. Preserve exact names of tools, queries, and key entities.

📌 OUTPUT FORMAT:
Construct the overall research approach: [One-sentence summary of the hierarchical strategy]
Manager Agent's Key Actions:
- [Action 1: e.g., Delegate task to Search Agent with specific query] → [Outcome/Insight]
- [Action 2: e.g., Receive report and issue final_answer] → [Final output]
Search Agent's Research Process:
1. [Tool/Action taken] → [Outcome, e.g., "Failed, triggered fallback to jina_search"]
2. [Tool/Action taken] → [Outcome, e.g., "Retrieved Discogs page, extracted album list"]
3. [Tool/Action taken] → [Outcome, e.g., "Finalized report with 3 albums, including Cantora 1 & 2"]
Potential Inconsistencies:
- [If any, describe the discrepancy, e.g., "Search Agent labeled Cantora 1 as a studio album, but Discogs categorizes it as a 'Compilation'. This may indicate a misclassification."]

---

## Example 1:
### Task
How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.

### Trajectory
(Manager Agent delegates task to Search Agent. Search Agent first tries `wikipedia_search`, fails, then uses `jina_search` and `crawl_pages` on Discogs. Search Agent returns a report listing 3 albums, including Cantora 1 & 2. Manager Agent directly outputs `final_answer(3)`.)

### Output
Construct the overall research approach: Manager Agent delegates detailed research to Search Agent, who employs a fallback strategy after initial search failure and compiles a report from authoritative sources.
Manager Agent's Key Actions:
- Delegated task to Search Agent with query: "Mercedes Sosa discography 2000-2009 studio albums" → Received detailed report listing 3 albums.
- Issued final_answer(3) based on Search Agent's report → Output: 3.
Search Agent's Research Process:
1. Called `wikipedia_search('Mercedes Sosa discography')` → No page found, triggered fallback.
2. Called `wikipedia_search('Mercedes Sosa')` → Retrieved main page, located "Studio albums" section.
3. Called `jina_search('Mercedes Sosa studio albums 2000-2009')` → Retrieved list of authoritative sources (Discogs, AllMusic, etc.).
4. Called `crawl_pages('https://en.wikipedia.org/wiki/Mercedes_Sosa')` → Extracted: Misa Criolla (2000), Corazón Libre (2005), Cantora 1 (2009).
5. Called `crawl_pages('https://www.discogs.com/artist/333361-Mercedes-Sosa')` → Cross-verified album names and years, but Discogs lists "Cantora 1" under "Compilations".
6. Called `final_answer(...)` → Reported 3 albums, including Cantora 1 (2009).
Potential Inconsistencies:
- The Search Agent's report classifies "Cantora 1" (2009) as a studio album. However, Discogs (a primary source consulted) categorizes releases like "Cantora 1" under "Compilations", suggesting it may not be a traditional studio album. The Manager Agent's final answer of "3" may be inaccurate if "Cantora 1" is excluded.

---

## YOUR TURN:
### Task
{task}

### Trajectory
{trajectory}

### Output
"""

PROMPTS["project_insights_system_prompt"]="""
You are a thoughtful and context-aware agent. You will be given a specific agent **role** and a set of **general insights** that apply to all roles. 
Your task is to **adapt these general insights** into **personalized insights tailored to the given role**, helping the agent perform more effectively.
Make sure your output aligns with the role's background, responsibilities, and point of view.

NOTE - Your output should follow the below format:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

PROMPTS["project_insights_user_prompt"]="""
### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""

PROMPTS["merge_rules_user_prompt"]="""
## Here are the current insights that need to be merged:
{current_rules}

## Please consolidate and rewrite them into **no more than {limited_number} refined insights**.

As the summarizing agent, remove redundancies, combine similar ideas, and ensure clarity.

Your output:
"""

PROMPTS["merge_rules_system_prompt"]="""
You are an agent skilled at summarizing and distilling insights. You are given a list of insights that were previously extracted from similar tasks. These insights may contain redundancy or overlap.

Your job is to **merge and consolidate similar insights**, and output a refined version that is **clear, actionable, and concise**.

NOTE:
- All merged insights **must be based strictly on the given inputs**. You are **not allowed to make up** or infer any new information.
- The output should be easy to read and follow.

📝 Output Format:
- Start your response directly with the numbered list, no preamble or explanations.
- Each insight should be a short sentence.
- Use the following format exactly:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

PROMPTS["finetune_insights_suffix"]={
    "full":"""
Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.
""",
    "not_full":""""""
}

PROMPTS["format_rules_operation_template"]="""
<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation.
"""


PROMPTS["critique_compare_rules_user_prompt"]="""
## Trial Task 1 (success):
{task1}
{task1_trajectory}

## Trial Task 2 (fail):
### Failed reason
{fail_reason}

### Trajectory
{task2}
{task2_trajectory}

## Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:
"""+PROMPTS["format_rules_operation_template"]

PROMPTS["critique_compare_rules_system_prompt"]="""
You are an advanced reasoning agent capable of deriving rules based on examples. You will be given two similar tasks: 
the first one is correct, and the second one is incorrect, The reason for failure has already been provided for the failed trajectory.

Requirements:
- Convert the reasons for failure into insights for future agents to reference, in order to avoid making the same mistakes.
- The insights you summarize must follow the "XXX, because XXX" format. They should not mention specific items but should instead extract general success principles applicable to similar tasks. These insights must be enlightening and provide guidance for future problems.  
"""

PROMPTS["critique_success_rules_user_prompt"]="""
## Requirements:  
- Avoid vague statements; ensure each insight has a clear causal relationship.  
- Focus only on strategies that apply to a broad range of scenarios rather than case-specific advice.  
- Keep the language concise and to the point, ensuring clarity and practical value.  
- The insights you summarize must follow the "XXX, because XXX" format. They should not mention specific items but should instead extract general success principles applicable to similar tasks. These insights must be enlightening and provide guidance for future problems.  

## Examples:  
- Eliminate unnecessary thinking, because focusing on core objectives improves execution efficiency.
  
## Here are the trials:
{success_history}

## Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:
"""+PROMPTS["format_rules_operation_template"]

PROMPTS["critique_success_rules_system_prompt"]="""
You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. 
You will be given successful tasks trials in which you were placed in a household environment and tasks to complete.
"""

PROMPTS["generative_task_system_prompt"]="""You are an agent designed to score the relevance between two pieces of text."""

PROMPTS["generative_task_user_prompt"]='''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''


