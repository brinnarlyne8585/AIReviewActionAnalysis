# 📁 _data Directory

This directory contains structured data collected from 16 AI-based code review actions.  
Each subdirectory (named after a GitHub repository, e.g., `anc95_ChatGPT-CodeReview`) follows a consistent internal structure.

Using `anc95_ChatGPT-CodeReview` as an example, each action folder includes:



## 📂 crawled_data/

Raw data crawled from GitHub.

- `repo_mentioning.csv`: Repositories using these actions  
- `repo_mentioning(pr>=50).csv`: Filtered version of `repo_mentioning.csv` containing only repositories with ≥50 PRs
- `pr_review_map.csv`: How many repos have actually left traces of the AI review action?  
- `valid_reviews.csv`: Only consider review comments in merged PRs (& before merged), and first in the conmunication thread, and where the primary language is English.
- `valid_reviews(diff_reshaped).csv`: Reshape the entire review diff for file-level review ation
- `valid_human_reviews.csv`: English Human Review in Merged PR (& before merged) During Action Activate, and Only consider the first comment in the conmunication thread
- `reviewed_file_versions.parquet`: Stores the file content (for both AI-generated and human comments) at the commit where the comment was made and at the PR merge commit.  This is used to analyze **post-review file changes**.

These data files correspond directly to the staged pipeline described in `statistics/Data_Collection.xlsx`.



## 📂 llm_input/

This folder contains standardized inputs for the LLM-assisted addressing analysis used in RQ2.

Preprocessed using `llm_analysis/organize_input_review.py`, based on the following mapping:

- `crawled_data/valid_reviews(diff_reshaped).csv` ⮕ `llm_input/valid_reviews(diff_reshaped)(llm_input)(consider_path).csv` (AI-generated comments)
- `crawled_data/valid_human_reviews.csv` ⮕ `llm_input/valid_human_reviews(llm_input)(consider_path).csv` (Human-written comments)


