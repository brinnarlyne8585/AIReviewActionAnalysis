# Perplexity-based LDA hyperparameter tuning module (Optuna-based).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from gensim.models.ldamodel import LdaModel
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
from address_feature_matrix.extract_topic_with_LDA import preprocess_comment

lemmatizer = WordNetLemmatizer()
project_vocab_tracker = defaultdict(Counter)


def get_project_unique_words() -> set:
    word_projects = defaultdict(set)
    for proj, counter in project_vocab_tracker.items():
        for word in counter:
            word_projects[word].add(proj)
    return {word for word, projects in word_projects.items() if len(projects) == 1}

def remove_project_specific_words(text: str, unique_words: set) -> str:
    return ' '.join([w for w in text.split() if w not in unique_words])

# === LDA perplexity evaluation. ===
def train_lda_multiple_runs(texts, k, alpha, beta, n_runs=5, passes=10):
    from gensim.corpora import Dictionary
    perplexities = []
    for _ in range(n_runs):
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            alpha=alpha,
            eta=beta,
            passes=passes,
            random_state=None,
            eval_every=None
        )
        log_perp = lda.log_perplexity(corpus)
        true_perp = np.exp(-log_perp)
        perplexities.append(true_perp)
    return np.mean(perplexities)

# === Optuna optimization objective function. ===
def objective(trial):
    k = trial.suggest_int("num_topics", 5, 10)
    alpha = trial.suggest_float("alpha", 0.001, 10.0, log=True)
    beta = trial.suggest_float("beta", 0.001, 10.0, log=True)
    try:
        score = train_lda_multiple_runs(all_texts, k, alpha, beta, n_runs=5)
        return score  # The objective is to minimize perplexity.
    except Exception as e:
        return float('inf')  # Return a large value in case of an error.

# === Main program entry point. ===
def run_lda_perplexity_optimization(input_file, apply_regex=True, remove_unique=False):
    df = pd.read_csv(input_file)
    docs = df["Cleaned_Body"].dropna().astype(str).tolist()
    ids = df["Comment_ID"].dropna().astype(str).tolist()

    global all_texts
    all_texts = [
        preprocess_comment(doc, comment_id=ids[i] if i < len(ids) else None,
                           apply_regex=apply_regex, remove_unique=remove_unique).split()
        for i, doc in enumerate(docs)
    ]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n🏆 Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    print(f"🔥 Best perplexity : {study.best_value:.4f}")



if __name__ == "__main__":

    run_lda_perplexity_optimization("_selected_reviews_with_features/reviews_contains_valid_by_LLM.csv")
