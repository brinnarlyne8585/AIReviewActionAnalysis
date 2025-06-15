# 🤖 llm_analysis

This module provides the **LLM-assisted framework** for assessing whether code review comments are **actionable** and whether they have been **addressed**.  




## 📂 Key Scripts

- `organize_input_review.py`:  
  Prepares post-review file change contexts for each of the 5,652 review comments in Table V.  

- `extract_suggestion_prompter_3.py`:  
  Prompt template for **Stage 1**, where we examined the comment text and the reviewed changes to determine whether a comment contained **valid (i.e., specific and actionable)** issues or suggestions.

- `analyze_addressed_prompter_4.py`:  
  Prompt template for **Stage 2**, which further analyzes subsequent file changes for comments marked as “Valid” in Stage 1.  
  The goal is to determine **to what extent** the identified suggestions were addressed (Fully, Partly, Not, or Not Enough Info).

- `pipeline_for_sample.py`:  
  Entry point for analyzing a **sampled and manually labeled subset** of the data, used for benchmarking LLM performance.

- `evaluation.py`:  
  Provides evaluation functions to assess model accuracy on the sampled data.

- `pipeline_for_main.py`:  
  Entry point for analyzing the **full set of 5,652 comments** with the full pipeline.




## 📂 input/

This folder contains the preprocessed review comment data used as input.

- `reviews(llm_input)(consider_path).csv`:  
  A consolidated file that merges LLM input data across actions.  
  It contains all **5,652 comments** used in our full-scale analysis (see Table V).




## 📂 output/

This folder stores all results from the LLM runs.

- `Suggestion_*.csv`:  
  Results from **Stage 1**, i.e., whether each comment was valid.

- `Addressed_*_based_Suggestion_*.csv`:  
  Results from **Stage 2**, i.e., whether and how the valid suggestions were addressed.

Each file name reflects the LLM model used (e.g., `openai-gpt-4.1`, `claude-3-7-sonnet`, `deepseek-v3`) and the parameter configuration.

