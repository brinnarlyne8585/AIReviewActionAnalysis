import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer
import shap
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2

from address_feature_matrix.build_features import FEATURE_GROUP_MAPPING

# --------------- Construct the tokenizer.  --------------- #
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer()
def lemmatize_tokenize(text):
    if not text or text.isspace():
        return []
    try:
        # Use NLTK for tokenization.
        tokens = tokenizer.tokenize(text)
        # Perform lemmatization on tokens and remove stopwords.
        lemmatized_tokens = [
            lemmatizer.lemmatize(token.lower())
            for token in tokens
            if token.lower() not in stop_words and token.isalpha()
        ]
        if not lemmatized_tokens:
            lemmatized_tokens.append("<EMPTY>")
        return lemmatized_tokens
    except IndexError:
        print(f"Error processing text: {text}")
        return []

def build_feature_union(enabled_feature_types: dict,
                        text_column: str = "Text") -> ColumnTransformer:
    """
    Construct a ColumnTransformer (feature union) supporting structured features and TF-IDF features (optional SelectKBest).

    :param enabled_feature_types: Configuration dictionary including tfidf, tfidf_select_k, tfidf_k, etc.
    :param text_column: Name of the text column
    :param tokenizer: External lemmatize_tokenize function can be passed in
    :return: ColumnTransformer object
    """
    transformer_list = []

    # ✅ Text features (TF-IDF or TF-IDF + SelectKBest).
    if enabled_feature_types.get("tfidf", False):
        print(f"🧮 Enable TF-IDF features.")
        normal_pipeline = TfidfVectorizer(
            tokenizer=lemmatize_tokenize,
            # min_df=3,
            # max_df=0.5,
            max_features=100,
        )
        select_k_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=lemmatize_tokenize)),
            ("select_k", SelectKBest(score_func=chi2, k=100))
        ])
        pipeline = select_k_pipeline
        transformer_list.append(("text", pipeline, text_column))
    else:
        print("⚠️ TF-IDF features not enabled.")

    # ✅ Structured features (always retained).
    transformer_list.append((
        "struct", FunctionTransformer(func=lambda x: np.vstack(x), validate=False), "structured_feature"
    ))

    return ColumnTransformer(transformer_list)


# --------------- Helper function to extract feature names. --------------- #
def extract_feature_names(pipeline, test_combined, text_column="Text", struct_feature_names=None):
    """
    Extract feature names from a trained pipeline, supporting cases with or without TF-IDF.

    Parameters:
    - pipeline: Trained model pipeline
    - test_combined: Test dataset
    - text_column: Name of the text feature column
    - struct_feature_names: List of structured feature names (optional)

    Returns:
    - all_feature_names: List of all feature names
    - has_tfidf: Whether TF-IDF features are included
    """
    all_feature_names = []

    # Check whether the pipeline contains text features (TF-IDF).
    has_tfidf = 'text' in pipeline.named_steps['features'].named_transformers_

    # 1. Retrieve TF-IDF feature names (if enabled).
    if has_tfidf:
        tfidf_vectorizer = pipeline.named_steps['features'].named_transformers_['text']
        try:
            tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        except:
            try:
                tfidf_feature_names = tfidf_vectorizer.get_feature_names()
            except:
                tfidf_features = tfidf_vectorizer.transform([test_combined[text_column].iloc[0]])
                tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]

        all_feature_names.extend(list(tfidf_feature_names))

    # 2. Retrieve structured feature names.
    if struct_feature_names is None:
        # If structured feature names are not provided, extract them from the features.
        struct_features_array = pipeline.named_steps['features'].named_transformers_['struct'].transform(
            test_combined['structured_feature'])
        struct_feature_names = [f"struct_{i}" for i in range(struct_features_array.shape[1])]

    all_feature_names.extend(struct_feature_names)

    return all_feature_names, has_tfidf


# --------------- Use SHAP.TreeExplainer for interpretation.  --------------- #
def explain_with_TreeExplainer(pipeline,
                               train_combined,
                               test_combined,
                               text_column="Text",
                               struct_feature_names: list = None):
    """
    Use SHAP's TreeExplainer to interpret a random forest model, supporting cases with or without TF-IDF features.

    Parameters:
    - pipeline: Trained pipeline model
    - test_combined: Test data containing both text and structured features
    - text_column: Name of the text column
    """

    # Extract feature names.
    all_feature_names, has_tfidf = extract_feature_names(
        pipeline, test_combined, text_column, struct_feature_names
    )

    # Process training data as background data.
    train_feature_list = []
    test_feature_list = []

    # 1. Retrieve TF-IDF features (if enabled).
    if has_tfidf:
        print("📊 Process TF-IDF features...")
        tfidf_vectorizer = pipeline.named_steps['features'].named_transformers_['text']
        # TF-IDF features of the test data.
        test_tfidf_features = tfidf_vectorizer.transform(test_combined[text_column]).toarray()
        test_feature_list.append(test_tfidf_features)
        # TF-IDF features of the train data.
        train_tfidf_features = tfidf_vectorizer.transform(train_combined[text_column]).toarray()
        train_feature_list.append(train_tfidf_features)
    else:
        print("📊 TF-IDF features not enabled, skipping...")

    # 2. Retrieve structured features.
    train_struct_features = np.array(train_combined['structured_feature'].tolist())
    train_feature_list.append(train_struct_features)
    test_struct_features = np.array(test_combined['structured_feature'].tolist())
    test_feature_list.append(test_struct_features)

    # 3. Horizontally stack all features.
    train_feature_arr = np.hstack(train_feature_list).astype(np.float64)
    test_feature_arr = np.hstack(test_feature_list).astype(np.float64)

    # 4. Retrieve the trained model.
    model = pipeline.named_steps['classifier']

    # 5. Use SHAP to interpret the model, setting background data.
    explainer = shap.TreeExplainer(model,
                                   data=train_feature_arr,
                                   feature_perturbation="interventional",
                                   )
    shap_values = explainer.shap_values(test_feature_arr,
                                        check_additivity=False,
                                        )
    verify_shap_additivity(model, explainer, shap_values, test_feature_arr)

    # 6. Visualization.
    print("📊 SHAP Top Feature Importance:")
    for i, class_shap_values in enumerate(shap_values):
        if i == 0:
            continue

        print(f"Class {i}:")
        shap.summary_plot(class_shap_values, test_feature_arr, feature_names=all_feature_names, show=False)
        plt.savefig("./explain_output/shap_summary_plot.png", bbox_inches="tight")
        plt.close()

        analyze_shap_feature_statistics(class_shap_values, test_feature_arr, all_feature_names)

        abs_mean_shap_values, feature_stats_sorted = analyze_shap_feature_statistics_by_group(
            class_shap_values, test_feature_arr, all_feature_names
        )
        summarize_shap_by_group(abs_mean_shap_values, all_feature_names)


def verify_shap_additivity(model, explainer, shap_values, test_feature_arr, class_index=1, verbose=True):
    """
    Validate SHAP additivity: expected_value + shap_values ≈ model.predict_proba
    - model: sklearn RandomForestClassifier model
    - explainer: shap.TreeExplainer instance
    - shap_values: shap_values[class_index]
    - test_feature_arr: Feature array of the test data
    - class_index: Class index to validate (default is class=1 for binary classification)
    - verbose: Whether to print detailed error statistics
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[class_index]
        expected_value = explainer.expected_value[class_index]
        predicted = model.predict_proba(test_feature_arr)[:, class_index]
    else:
        expected_value = explainer.expected_value
        predicted = model.predict(test_feature_arr)

    reconstructed = expected_value + shap_values.sum(axis=1)
    abs_errors = np.abs(reconstructed - predicted)

    if verbose:
        print("🧮 SHAP Additivity Error Statistics (Unit: Absolute Difference)")
        print(f"- Minimum error: {abs_errors.min():.6e}")
        print(f"- Maximum error: {abs_errors.max():.6e}")
        print(f"- Mean error   : {abs_errors.mean():.6e}")
        print(f"- Std. dev.    : {abs_errors.std():.6e}")

    return abs_errors


def analyze_shap_feature_statistics(class_shap_values, test_feature_arr, all_feature_names):
    # 7. Calculate and print feature importance statistics.
    feature_stats = []
    abs_mean_shap_values = np.abs(class_shap_values).mean(axis=0)

    for j, feature in enumerate(all_feature_names):
        abs_mean = abs_mean_shap_values[j]
        # Calculate Pearson correlation coefficient
        corr, _ = pearsonr(test_feature_arr[:, j], class_shap_values[:, j])
        # Append feature, abs_mean, and correlation to the list
        feature_stats.append((feature, abs_mean, corr))

    # Sort by abs_mean in descending order
    feature_stats_sorted = sorted(feature_stats, key=lambda x: x[1], reverse=True)
    # Print the sorted results
    for feature, abs_mean, corr in feature_stats_sorted:
        print(f"{feature}:\t{abs_mean:.4f}\t{corr:.4f}")

def analyze_shap_feature_statistics_by_group(class_shap_values, test_feature_arr, all_feature_names):
    """
    Output the average SHAP value and Pearson correlation coefficient for each feature, grouped by feature category.
    Returns: abs_mean_shap_values, feature_stats_sorted
    """

    print("\n📊 Feature SHAP Importance by Group:")
    abs_mean_shap_values = np.abs(class_shap_values).mean(axis=0)
    feature_stats = []
    feature_stats_dict = {}

    for j, feature in enumerate(all_feature_names):
        abs_mean = abs_mean_shap_values[j]
        corr, _ = pearsonr(test_feature_arr[:, j], class_shap_values[:, j])
        feature_stats.append((feature, abs_mean, corr))
        feature_stats_dict[feature] = {"mean_abs_shap": abs_mean, "pearson_corr": corr}

    # Output detailed values and plots by group (sorted by SHAP value, using \t for easy copying into spreadsheets).
    for group, features in FEATURE_GROUP_MAPPING.items():
        print(f"\n🔹 Group: {group}")
        group_items = []
        for f in features:
            if f in feature_stats_dict:
                stat = feature_stats_dict[f]
                group_items.append((f, stat['mean_abs_shap'], stat['pearson_corr']))

        for f, shap_val, corr in sorted(group_items, key=lambda x: x[1], reverse=True):
            print(f"{f}\t{shap_val:.4f}\t{corr:.4f}")

        # Drawing
        if group_items:
            feat_names, shap_vals = zip(*[(f, s) for f, s, _ in sorted(group_items, key=lambda x: x[1], reverse=True)])
            plt.figure(figsize=(8, 0.4 * len(feat_names)))
            plt.barh(feat_names, shap_vals)
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"SHAP Importance: {group}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"./explain_output/shap_bar_{group}.png")
            plt.close()

    global_avg = abs_mean_shap_values.mean()
    print(f"\n🌍 Global Average Feature Importance (mean(|SHAP|) over all features): {global_avg:.4f}")

    return abs_mean_shap_values, sorted(feature_stats, key=lambda x: x[1], reverse=True)


def summarize_shap_by_group(abs_mean_shap_values, all_feature_names):
    """
    Calculate the sum and mean of SHAP values for each feature category and generate plots.
    """

    group_importance_avg = {}
    group_importance_sum = {}

    for group, features in FEATURE_GROUP_MAPPING.items():
        group_indices = [i for i, name in enumerate(all_feature_names) if name in features]
        if not group_indices:
            continue
        group_vals = abs_mean_shap_values[group_indices]
        group_importance_avg[group] = np.mean(group_vals)
        group_importance_sum[group] = np.sum(group_vals)

    print("\n📊 Group SHAP (Average):")
    for k, v in sorted(group_importance_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}\t{v:.4f}")

    print("\n📊 Group SHAP (Sum):")
    for k, v in sorted(group_importance_sum.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}\t{v:.4f}")

    # Plot: Mean.
    plt.figure(figsize=(8, 5))
    groups = list(group_importance_avg.keys())
    avg_vals = [group_importance_avg[g] for g in groups]
    plt.barh(groups, avg_vals)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Average SHAP Importance by Feature Group")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.savefig("./explain_output/shap_group_avg_importance.png")
    plt.close()

    # Plot: Sum.
    plt.figure(figsize=(8, 5))
    sum_vals = [group_importance_sum[g] for g in groups]
    plt.barh(groups, sum_vals)
    plt.xlabel("Total Absolute SHAP Value")
    plt.title("Total SHAP Importance by Feature Group")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.savefig("./explain_output/shap_group_sum_importance.png")
    plt.close()





# --------------- Interpret the model using the built-in feature_importances_ attribute of RandomForestClassifier --------------- #
def explain_with_feature_importance_from_tree(pipeline,
                                            test_combined,
                                            top_k: int = 30,
                                            text_column: str = "Text",
                                            struct_feature_names: list = None):
    """
    Use the feature_importances_ attribute of RandomForestClassifier for global feature interpretation, and output the top_k feature importance rankings and bar chart.
    Supports cases with or without TF-IDF features.

    :param pipeline: Trained pipeline object
    :param test_combined: Test data containing structured features and text column
    :param top_k: Number of top important features to display (default 30)
    :param text_column: Name of the text column (default "Text")
    """

    # 1. Retrieve the classifier model.
    model = pipeline.named_steps['classifier']

    # 2. Get feature names
    all_feature_names, _ = extract_feature_names(
        pipeline, test_combined, text_column, struct_feature_names
    )

    # 3. Get feature importances
    importances = model.feature_importances_
    feature_importance_list = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)

    # 4. Print top_k features
    print(f"🌲 Top {top_k} Feature Importances (RandomForest):")
    for name, score in feature_importance_list[:min(top_k, len(feature_importance_list))]:
        print(f"{name}: {score:.4f}")

    # 5. Visualize bar chart
    names, scores = zip(*feature_importance_list[:min(top_k, len(feature_importance_list))])
    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], scores[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_k} Feature Importances from Random Forest")
    plt.tight_layout()
    plt.show()
