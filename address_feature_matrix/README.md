# 📊 address_feature_matrix

This module includes data and scripts used in **RQ3** to analyze which factors influence whether code review comments lead to code changes.

To overcome the inherent lack of interpretability in LLMs, we engineered a structured feature set covering various dimensions that potentially affect response behavior. We then trained a Random Forest classifier to fit the LLM-derived addressing results and used SHAP analysis to interpret the influence and directionality of each feature on the model’s predictions.



## 📂 Data Files

- `_all_review_with_features/all_reviews_with_features.csv`:  
  Contains the feature set (excluding LDA topics) for all **5,652 comments** involved in the addressing analysis.  

- `_selected_reviews_with_features/reviews_contains_valid_by_LLM.csv`:  
  Contains the feature set (excluding LDA topics) for the filtered **3,879 valid comments** that have a definitive addressing label.  
  These are the comments selected for the final factor analysis in RQ3.  
  We exclude comments that are either non-valid (per Stage 1) or whose address status is "Not Enough Info" (per Stage 2).



## 📂 Topic Modeling Scripts

- `lda_optuna_perplexity.py`:  
  Following the method proposed in [C. Treude et al., MSR 2019](https://ieeexplore.ieee.org/document/8816780), performs hyperparameter tuning to find the best LDA topic model parameters for the 3,879 valid comments.
- `extract_topic_with_LDA.py`:  
  Applies the best parameters to train an LDA model and extract topic distributions for each comment.

- `describe_topic.py`:  
  Uses an LLM to generate **interpretable, human-readable topic descriptions** for each discovered LDA topic.



## 📂 Feature Selection and Modeling

- `select_feature_with_autospearman.py`:  
  To further address potential multicollinearity, we applied **AutoSpearman** correlation analysis ([J. Jiarpakdee et al., ICSME 2018](https://ieeexplore.ieee.org/document/8530020)) and removed 9 redundant features, yielding a final set of 36 features.

- `analyze_feature_importance_with_rf.py`:  
  Trains a **Random Forest classifier** on the 3,879 labeled valid comments and uses **SHAP** to explain the impact of each feature on its prediction outcome.
