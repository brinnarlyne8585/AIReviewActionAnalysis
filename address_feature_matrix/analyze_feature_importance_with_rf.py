import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import plot_tree


from address_feature_matrix.classifier_common import (
    build_feature_union,
    explain_with_feature_importance_from_tree,
    explain_with_TreeExplainer,
)
from address_feature_matrix.build_features import assemble_feature_matrix


# -------------------------------- Main training and evaluation function. -------------------------------- #
def train_rf_with_tfidf_and_structured(
        df: pd.DataFrame,
        enabled_feature_types: dict,
        text_column: str = "Text",
        label_column: str = "Label",
        test_size: float = 0.2,
        random_state: int = 42,
        explain_method: str = None):

    # Split the data.
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_column])
    y_train = train_df[label_column].values
    y_test = test_df[label_column].values

    # Feature construction.
    X_train_struct_df = assemble_feature_matrix(train_df, enabled_feature_types)
    X_test_struct_df = assemble_feature_matrix(test_df, enabled_feature_types)
    X_train_struct = X_train_struct_df.values
    X_test_struct = X_test_struct_df.values
    struct_feature_names = list(X_train_struct_df.columns)

    print(f"{len(struct_feature_names)}: {struct_feature_names}")

    train_combined = pd.DataFrame({
        text_column: train_df[text_column].tolist(),
        "structured_feature": list(X_train_struct)
    })
    test_combined = pd.DataFrame({
        text_column: test_df[text_column].tolist(),
        "structured_feature": list(X_test_struct)
    })

    feature_union = build_feature_union(
        enabled_feature_types=enabled_feature_types,
        text_column=text_column,
    )

    # Init pipeline
    pipeline = Pipeline([
        ("features", feature_union),
        ("classifier", RandomForestClassifier(random_state=random_state))
    ])

    pipeline.fit(train_combined, y_train)

    # Prediction evaluation.
    y_pred = pipeline.predict(test_combined)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"🎯 Final Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    # Model interpretation.
    explain_model(
        pipeline,
        explain_method,
        train_combined,
        test_combined,
        text_column,
        struct_feature_names,
    )

    return pipeline, acc, f1



# --------------- Model interpretation  --------------- #
def explain_model(pipeline,
                  method,
                  train_combined,
                  test_combined,
                  text_column,
                  struct_feature_names):
    """
    Model interpretation interface function, supporting multiple interpretation methods and optional tree structure visualization.

    :param pipeline: Trained pipeline model
    :param method: Interpretation method ["rf", "shap.TreeExplainer", "none"]
    :param test_combined: Test data (including text and structured features)
    :param text_column: Name of the text column
    :param struct_feature_names: List of structured feature names
    :param visualize_tree: Whether to visualize one of the trees in the random forest
    """

    if method == "rf":
        explain_with_feature_importance_from_tree(
            pipeline=pipeline,
            test_combined=test_combined,
            top_k=200,
            text_column=text_column,
            struct_feature_names=struct_feature_names
        )
    elif method == "shap.TreeExplainer":
        explain_with_TreeExplainer(
            pipeline=pipeline,
            train_combined=train_combined,
            test_combined=test_combined,
            text_column=text_column,
            struct_feature_names=struct_feature_names
        )
    elif method is None or method.lower() == "none":
        print("ℹ️ Skip explain")
    else:
        print(f"⚠️ Unsupported interpretation method: {method}")



if __name__ == "__main__":

    enabled_config = {
        "repo": True,
        "modification": True,
        "textual": True, # textual meta features（e.g. text length etc.）
        "file": True,
        "action": True,
        "topic": True,  # ✅ Enable topic features.
        "tfidf": False,
    }


    feature_df = pd.read_csv("_selected_reviews_with_features/reviews_contains_valid_by_LLM_lda_topics_selected.csv")
    label_df = pd.read_csv("labels/all_reviews_with_LLM_labels.csv")

    # Merge data.
    df = pd.merge(feature_df, label_df[["Comment_ID", "Change_Label"]], on="Comment_ID", how="inner")

    # Perform training.
    pipeline, acc, f1 = train_rf_with_tfidf_and_structured(
        df=df,
        enabled_feature_types=enabled_config,
        text_column="Cleaned_Body",
        label_column="Change_Label",
        explain_method="shap.TreeExplainer",
    )
