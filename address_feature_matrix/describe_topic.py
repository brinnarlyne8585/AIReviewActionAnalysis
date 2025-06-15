import json
import re

from config import BASE_DIR
import pandas as pd
from typing import Dict, List
from call_llm import get_model_function

# ==== Parameter settings. ====
package_path = f"{BASE_DIR}/address_feature_matrix/_selected_reviews_with_features"
file_path = f"{package_path}/reviews_contains_valid_by_LLM_lda_topics_merged_selected.csv"
output_path = f"{package_path}/topic_summaries_gpt4o_from_lda(gpt).csv"
llm_model = "openai-gpt-4.1"
top_n = 10

# ==== LLM function. ====
response_function = get_model_function(llm_model, input_type="prompt")

# ==== Load data. ====
df = pd.read_csv(file_path)
df = df.dropna(subset=["Cleaned_Body"])

topic_prob_cols = [f"LDA_Topic_Prob_{i}" for i in range(6)]
sampled_topic_comment_map: Dict[int, List[str]] = {}

def remove_code_blocks(comment: str) -> str:
    """
    Remove Markdown code snippets from comments:
    - Multi-line code blocks ```...```
    - Single-line code snippets `...`
    """
    # Remove multi-line code blocks ```...```
    comment = re.sub(r'```[\s\S]*?```', ' ', comment)
    # Remove single-line inline code. `...`
    comment = re.sub(r'`[^`]+`', ' ', comment)
    # Merge redundant whitespace.
    comment = re.sub(r'\s+', ' ', comment)
    return comment.strip()

# ==== Retrieve the top-n comments for each topic. ====
for topic_id in range(6):
    df_topic = df[["Cleaned_Body", topic_prob_cols[topic_id]]].dropna()
    df_topic = df_topic.rename(columns={topic_prob_cols[topic_id]: "topic_prob"})
    top_comments = df_topic.sort_values("topic_prob", ascending=False).head(top_n)["Cleaned_Body"].tolist()
    cleaned_comments = [remove_code_blocks(c) for c in top_comments]
    sampled_topic_comment_map[topic_id] = cleaned_comments


# ==== Construct the prompt. ====
def build_prompt(comments: List[str]) -> str:
    comment_list = "\n".join(f"Comment-{i+1}: {json.dumps(c, ensure_ascii=False)}" for i, c in enumerate(comments))
    return f"""You are a software engineering and NLP expert. Below are 10 representative code review comments that belong to the same latent topic identified by LDA topic modeling.
Your task is to analyze these comments and generate a short, specific, and distinctive topic label that captures the shared theme across them.

Guidelines:
- The label should reflect the main technical concern, bug pattern, or improvement target discussed in the comments.
- Keep it concise (2–5 words) and highly specific—avoid vague terms like "general feedback" or "common issues".
- Prefer actionable or semantic terms developers would recognize, such as:
    - Null check handling
    - Boundary condition fix
    - Logging verbosity level
- Provide a short explanation, summarizing the shared focus of these comments.

Here are the representative comments:
{comment_list}

Output format:
Topic Label: 
Explanation: 

Output:"""


# ==== Invoke the LLM & save the results. ====
results = []

for topic_id, comments in sampled_topic_comment_map.items():
    prompt = build_prompt(comments)
    try:
        response = response_function(prompt).strip()
        label, explanation = "", ""
        if "Topic Label:" in response and "Explanation:" in response:
            label_part = response.split("Topic Label:")[-1].split("Explanation:")[0].strip()
            explanation_part = response.split("Explanation:")[-1].strip()
            label = label_part
            explanation = explanation_part
        else:
            label = "[PARSE ERROR]"
            explanation = response
    except Exception as e:
        label = "[ERROR]"
        explanation = str(e)

    results.append({
        "LDA_Topic_Label": topic_id,
        "Topic_Label": label,
        "Explanation": explanation,
        "Sampled_Comments": "\n-----\n".join(comments)
    })

# ==== Write to CSV. ====
df_out = pd.DataFrame(results)
df_out.to_csv(output_path, index=False)
print(f"[✔] Saved to {output_path}")
