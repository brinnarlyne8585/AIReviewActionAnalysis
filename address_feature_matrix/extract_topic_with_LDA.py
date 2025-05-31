from gensim.corpora import Dictionary
import os
import re
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis
import matplotlib
import matplotlib.pyplot as plt
import spacy
from nltk.stem import WordNetLemmatizer
import pyLDAvis_sklearn_model as sklearn_lda_vis

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

lemmatizer = WordNetLemmatizer()
cached_preprocessed_docs = []
project_vocab_tracker = defaultdict(Counter)

nlp = spacy.load("en_core_web_sm")
def preprocess_comment(
    comment: str,
    comment_id: Optional[str] = None,
    apply_regex: bool = True,
    remove_unique: bool = False
) -> str:
    # Remove Markdown code blocks（```...``` and `...`）
    if apply_regex:
        comment = re.sub(r'```[\s\S]*?```', ' ', comment)
        comment = re.sub(r'`[^`]+`', ' ', comment)

    # Use spaCy for tokenization and lemmatization.
    doc = nlp(comment)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

    # Project-specific vocabulary tracking (retain original functionality).
    if comment_id:
        project = extract_owner_repo(comment_id)
        project_vocab_tracker[project].update(tokens)

    # Remove project-specific terms (retain original functionality)
    processed_text = ' '.join(tokens)
    if remove_unique:
        unique_words = get_project_unique_words()
        processed_text = remove_project_specific_words(processed_text, unique_words)

    return processed_text

def extract_repo_info(url):
    """
    Extract the owner and name of the repository from Comment_URL.
    """
    try:
        # E.g.: https://api.github.com/repos/owner/repo/pulls/comments/comment_id
        parts = url.split("/")
        if len(parts) < 6:
            return None, None
        owner = parts[4]
        repo = parts[5]
        return owner, repo
    except Exception as e:
        print(f"[ERROR] Failed to extract repo info from URL: {url} - {e}")
        return None, None

def extract_owner_repo(url):
    owner, repo = extract_repo_info(url)
    return f"{owner}/{repo}"

def get_project_unique_words() -> set:
    word_projects = defaultdict(set)
    for proj, counter in project_vocab_tracker.items():
        for word in counter:
            word_projects[word].add(proj)
    return {word for word, projects in word_projects.items() if len(projects) == 1}


def remove_project_specific_words(text: str, unique_words: set) -> str:
    return ' '.join([w for w in text.split() if w not in unique_words])


class LDATopicModeler:
    def __init__(
        self,
        n_topics: int = 10,
        random_seed: int = 42,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        lda_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "./lda_output",
        apply_regex: bool = True,
        remove_unique: bool = False
    ):
        self.n_topics = n_topics
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.apply_regex = apply_regex
        self.remove_unique = remove_unique

        os.makedirs(output_dir, exist_ok=True)

        default_vectorizer_params = {
            "stop_words": "english",
            "min_df": 3,
            "max_df": 0.5
        }
        if vectorizer_params:
            default_vectorizer_params.update(vectorizer_params)
        self.vectorizer_params = default_vectorizer_params

        self.vectorizer = CountVectorizer(**self.vectorizer_params)

        lda_defaults = {"learning_method": "batch"}

        if lda_params:
            lda_defaults.update(lda_params)
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_seed,
            **lda_defaults
        )

        self.doc_topic_dist = None
        self.doc_topic_label = None
        self.documents = None

    def fit_transform(self, documents: List[str], comment_ids: Optional[List[str]] = None) -> np.ndarray:
        global cached_preprocessed_docs
        if not cached_preprocessed_docs:
            cached_preprocessed_docs = [
                preprocess_comment(doc, comment_id=comment_ids[i] if comment_ids else None, apply_regex=self.apply_regex, remove_unique=self.remove_unique)
                for i, doc in enumerate(documents)
            ]
        self.documents = cached_preprocessed_docs
        dtm = self.vectorizer.fit_transform(self.documents)
        self.doc_topic_dist = self.lda_model.fit_transform(dtm)
        self.doc_topic_label = np.argmax(self.doc_topic_dist, axis=1)
        return self.doc_topic_dist

    def print_top_words(self, n_words: int = 10):
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_indices]
            print(f"\n🔹 Topc {topic_idx}: {' '.join(top_words)}")

    def get_top_docs_per_topic(self, docs: List[str], top_k: int = 3) -> Dict[int, List[str]]:
        top_docs = defaultdict(list)
        for topic_id in range(self.n_topics):
            topic_probs = self.doc_topic_dist[:, topic_id]
            top_indices = topic_probs.argsort()[::-1][:top_k]
            top_docs[topic_id] = [docs[i] for i in top_indices]
        return top_docs

    def compute_coherence_score(self) -> float:
        texts = [doc.split() for doc in self.documents]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_topics = []
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_weights in self.lda_model.components_:
            top_indices = topic_weights.argsort()[:-11:-1]
            lda_topics.append([feature_names[i] for i in top_indices])
        coherence_model = CoherenceModel(topics=lda_topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"\n🧪 Topic coherence (c_v): {coherence_score:.4f}")
        return coherence_score

    def visualize_with_pyldavis(self):
        dtm = self.vectorizer.transform(self.documents)
        vis_data = sklearn_lda_vis.prepare(self.lda_model, dtm, self.vectorizer)
        output_path = os.path.join(self.output_dir, "lda_visualization.html")
        pyLDAvis.save_html(vis_data, output_path)
        print(f"📊 Topic visualization has been saved to {output_path}")

    def add_topics_to_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_path: str,
        topic_label_column: str = "LDA_Topic_Label",
        add_probabilities: bool = False
    ) -> pd.DataFrame:
        non_null_mask = df[text_column].notna()
        df.loc[non_null_mask, topic_label_column] = self.doc_topic_label

        if add_probabilities:
            prob_cols = [f"LDA_Topic_Prob_{i}" for i in range(self.n_topics)]
            for col in prob_cols:
                df[col] = np.nan
            for idx, probs in enumerate(self.doc_topic_dist):
                row_idx = non_null_mask.index[idx]
                for topic_id, prob in enumerate(probs):
                    df.at[row_idx, f"LDA_Topic_Prob_{topic_id}"] = prob

        df.to_csv(output_path, index=False)
        print(f"Topic assignment results saved to {output_path}")
        return df


def process_reviews_with_lda(
    input_file: str,
    output_file: str,
    text_column: str = "Cleaned_Body",
    id_column: str = "Comment_ID",
    n_topics: int = 10,
    random_seed: int = 42,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    lda_params: Optional[Dict[str, Any]] = None,
    add_probabilities: bool = False,
    apply_regex: bool = True,
    remove_unique: bool = False
):
    df = pd.read_csv(input_file)
    docs = df[text_column].dropna().astype(str).tolist()
    ids = df[id_column].dropna().astype(str).tolist()

    lda_modeler = LDATopicModeler(
        n_topics=n_topics,
        random_seed=random_seed,
        vectorizer_params=vectorizer_params,
        lda_params=lda_params,
        apply_regex=apply_regex,
        remove_unique=remove_unique
    )

    lda_modeler.fit_transform(docs, comment_ids=ids)
    lda_modeler.print_top_words(n_words=10)
    lda_modeler.compute_coherence_score()
    lda_modeler.visualize_with_pyldavis()

    df = lda_modeler.add_topics_to_dataframe(
        df=df,
        text_column=text_column,
        output_path=output_file,
        add_probabilities=add_probabilities
    )

    return df


TOPIC_NUMBER = 6
if __name__ == "__main__":

    input_file = "_selected_reviews_with_features/reviews_contains_valid_by_LLM.csv"

    lda_config = {
        "n_topics": 6,
        "lda_params": {
            "doc_topic_prior": 0.09422305476119705,
            "topic_word_prior": 0.18259251524681178,
        },
    }

    output_file = input_file.replace(".csv", "_lda_topics.csv")
    process_reviews_with_lda(
        input_file=input_file,
        output_file=output_file,
        n_topics=lda_config["n_topics"],
        lda_params=lda_config["lda_params"],
        add_probabilities=True,
        apply_regex = True,
        remove_unique=True,
    )


