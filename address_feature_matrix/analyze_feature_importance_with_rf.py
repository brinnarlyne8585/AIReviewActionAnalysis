import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import FunctionTransformer
import shap
from scipy.stats import pearsonr

from address_feature_matrix.build_features import assemble_feature_matrix, FEATURE_GROUP_MAPPING, map_model

# --------------- Tokenizer --------------- #
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer()


def lemmatize_tokenize(text):
    if not text or (isinstance(text, str) and text.isspace()):
        return []
    try:
        if not isinstance(text, str):
            text = str(text)
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
    except Exception as e:
        print(f"Error processing text: {text}, error: {e}")
        return []


def build_feature_union(enabled_feature_types: dict,
                        text_column: str = "Text") -> ColumnTransformer:
    """
    Build a FeatureUnion to support both structured and TF-IDF features.
    """
    transformer_list = []

    # ✅ # Text features (TF-IDF)
    if enabled_feature_types.get("tfidf", False):
        print(f"🧮 Enable TF-IDF features.")
        normal_pipeline = TfidfVectorizer(
            tokenizer=lemmatize_tokenize,
            max_features=100,
        )
        select_k_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=lemmatize_tokenize)),
            ("select_k", SelectKBest(score_func=chi2, k=100))
        ])
        pipeline = select_k_pipeline
        transformer_list.append(("text", pipeline, text_column))
    else:
        print("⚠️ Disable TF-IDF features.")

    # ✅ Structured features
    transformer_list.append((
        "struct", FunctionTransformer(func=lambda x: np.vstack(x), validate=False), "structured_feature"
    ))

    return ColumnTransformer(transformer_list)


def extract_feature_names(pipeline, test_combined, text_column="Text", struct_feature_names=None):
    """
    Get feature names from the pipeline.
    """
    all_feature_names = []

    # Check if TF-IDF features are enabled.
    has_tfidf = 'text' in pipeline.named_steps['features'].named_transformers_

    # 1. Get TF-IDF feature names.
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

    # 2. Obtain structured feature names
    if struct_feature_names is None:
        struct_features_array = pipeline.named_steps['features'].named_transformers_['struct'].transform(
            test_combined['structured_feature'])
        struct_feature_names = [f"struct_{i}" for i in range(struct_features_array.shape[1])]

    all_feature_names.extend(struct_feature_names)

    return all_feature_names, has_tfidf


def analyze_shap_feature_statistics(class_shap_values, test_feature_arr, all_feature_names):
    """
    Calculate and print feature importance statistics (sorted by abs_mean descending) with ranking.
    """
    feature_stats = []
    abs_mean_shap_values = np.abs(class_shap_values).mean(axis=0)

    for j, feature in enumerate(all_feature_names):
        abs_mean = abs_mean_shap_values[j]
        # Compute Pearson correlation
        if test_feature_arr.shape[1] > j:
            try:
                corr, _ = pearsonr(test_feature_arr[:, j], class_shap_values[:, j])
            except:
                corr = 0
        else:
            corr = 0
        feature_stats.append((feature, abs_mean, corr))

    # Sort by mean absolute SHAP value in descending order
    feature_stats_sorted = sorted(feature_stats, key=lambda x: x[1], reverse=True)

    # Print table header
    print(f"Rank\tFeature\tSHAP\tPearsonr")

    # Calculate ranks
    rank_list = []
    feature_rank_dict = {}

    current_rank = 1
    for i, (feature, abs_mean, corr) in enumerate(feature_stats_sorted):
        # Handle ties: Assign same rank if difference from previous value is within epsilon.
        if i > 0 and abs(abs_mean - feature_stats_sorted[i - 1][1]) < 1e-9:
            rank_list.append(rank_list[-1])
        else:
            rank_list.append(i + 1)

        feature_rank_dict[feature] = rank_list[-1]

    for i, (feature, abs_mean, corr) in enumerate(feature_stats_sorted):
        print(f"{rank_list[i]}\t{feature}:\t{abs_mean:.4f}\t{corr:.4f}")

    return feature_rank_dict


def analyze_shap_feature_statistics_by_group(class_shap_values, test_feature_arr, all_feature_names, feature_rank_dict):
    """
    Print SHAP importance by feature group.
    """
    print("\n📊 Print SHAP importance by feature group:")
    abs_mean_shap_values = np.abs(class_shap_values).mean(axis=0)
    feature_stats_dict = {}

    for j, feature in enumerate(all_feature_names):
        abs_mean = abs_mean_shap_values[j]
        if test_feature_arr.shape[1] > j:
            corr, _ = pearsonr(test_feature_arr[:, j], class_shap_values[:, j])
        else:
            corr = 0
        feature_stats_dict[feature] = {"mean_abs_shap": abs_mean, "pearson_corr": corr}

    for group, features in FEATURE_GROUP_MAPPING.items():
        print(f"\n🔹 Group: {group}")
        group_items = []
        for f in features:
            if f in feature_stats_dict:
                stat = feature_stats_dict[f]
                # Get global feature rank.
                rank = feature_rank_dict.get(f, -1)
                group_items.append((f, stat['mean_abs_shap'], stat['pearson_corr'], rank))

        # Sort within groups
        sorted_group_items = sorted(group_items, key=lambda x: x[1], reverse=True)

        for f, shap_val, corr, rank in sorted_group_items:
            # Format: Feature Name \t Global Rank \t SHAP \t Pearson
            print(f"{f}\t{rank}\t{shap_val:.4f}\t{corr:.4f}")

    global_avg = abs_mean_shap_values.mean()
    print(f"\n🌍 Global Average Feature Importance (mean(|SHAP|)): {global_avg:.4f}")

    return abs_mean_shap_values


def summarize_shap_by_group(abs_mean_shap_values, all_feature_names):
    """
    Calculate the sum and mean of SHAP values for each feature group.
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

    print("\n📊 Group SHAP (Mean):")
    for k, v in sorted(group_importance_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}\t{v:.4f}")

    print("\n📊 Group SHAP (Sum):")
    for k, v in sorted(group_importance_sum.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}\t{v:.4f}")


# ========================================================================================================
# 5-Fold Cross-Validation Main Logic
# ========================================================================================================

def train_rf_with_tfidf_and_structured_5fold(
        df: pd.DataFrame,
        enabled_feature_types: dict,
        text_column: str = "Text",
        label_column: str = "Label",
        random_state: int = 42,
        explain_method: str = "shap.TreeExplainer"):
    # 1. Prepare Data
    # Only build features first, ensuring all samples have structured features
    full_struct_df = assemble_feature_matrix(df, enabled_feature_types)

    # Remove "Is_Human" feature (if it exists and is requested to be excluded)
    if Is_AI_Specific and "Is_Human" in full_struct_df.columns:
        print("ℹ️ 'Is_Human' feature excluded as requested.")
        full_struct_df = full_struct_df.drop(columns=["Is_Human"])

    full_struct_values = full_struct_df.values
    struct_feature_names = list(full_struct_df.columns)
    print(f"Number of Features: {len(struct_feature_names)}: {struct_feature_names}")

    # Combine text and structured features for easy splitting
    combined_data = pd.DataFrame({
        text_column: df[text_column].tolist(),
        "structured_feature": list(full_struct_values),
        label_column: df[label_column].tolist()
    })

    X = combined_data  # Contains text and structured_feature
    y = combined_data[label_column].values

    # 2. Initialize 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    accuracies = []
    f1_scores = []

    # Accumulate all data required for SHAP analysis
    all_shap_values_list = []  # Store SHAP values calculated for each fold
    all_test_features_list = []  # Store test set feature matrix (transformed) for each fold

    # Save feature names (assuming they are the same for each fold)
    final_all_feature_names = None

    print(f"\n🚀 Starting 5-Fold Cross-Validation...")

    fold_idx = 1
    for train_index, test_index in skf.split(X, y):
        print(f"\n--- Fold {fold_idx} ---")

        # Split Data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 3. Build Pipeline
        feature_union = build_feature_union(
            enabled_feature_types=enabled_feature_types,
            text_column=text_column,
        )

        pipeline = Pipeline([
            ("features", feature_union),
            ("classifier", RandomForestClassifier(
                random_state=random_state,
            ))
        ])

        # 4. Train
        pipeline.fit(X_train, y_train)

        # 5. Predict and Evaluate
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        accuracies.append(acc)
        f1_scores.append(f1)

        print(f"Fold {fold_idx} Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

        # 6. SHAP Explanation Data Collection (if needed)
        if explain_method == "shap.TreeExplainer":
            # Extract feature names
            all_feature_names, has_tfidf = extract_feature_names(
                pipeline, X_test, text_column, struct_feature_names
            )
            if final_all_feature_names is None:
                final_all_feature_names = all_feature_names

            # Convert test data to the feature matrix format expected by the model
            # Need to handle TF-IDF and structured features separately
            test_feature_list = []
            train_feature_list = []  # Used for background

            if has_tfidf:
                tfidf_vectorizer = pipeline.named_steps['features'].named_transformers_['text']
                test_tfidf = tfidf_vectorizer.transform(X_test[text_column]).toarray()
                train_tfidf = tfidf_vectorizer.transform(X_train[text_column]).toarray()
                test_feature_list.append(test_tfidf)
                train_feature_list.append(train_tfidf)

            test_struct = np.array(X_test['structured_feature'].tolist())
            train_struct = np.array(X_train['structured_feature'].tolist())
            test_feature_list.append(test_struct)
            train_feature_list.append(train_struct)

            test_feature_arr = np.hstack(test_feature_list).astype(np.float64)
            train_feature_arr = np.hstack(train_feature_list).astype(np.float64)

            # Calculate SHAP
            model = pipeline.named_steps['classifier']
            explainer = shap.TreeExplainer(model, data=train_feature_arr, feature_perturbation="interventional")

            # SHAP values for class 1
            shap_values = explainer.shap_values(test_feature_arr, check_additivity=False)

            # Only collect SHAP values for Class 1 (Binary classification usually focuses on the positive class)
            if isinstance(shap_values, list):
                class_1_shap = shap_values[1]
            else:
                class_1_shap = shap_values

            all_shap_values_list.append(class_1_shap)
            all_test_features_list.append(test_feature_arr)

        fold_idx += 1

    # 7. Report Average Metrics
    avg_acc = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)

    print("\n" + "=" * 50)
    print(f"📊 5-Fold Cross-Validation Average Results:")
    print(f"🎯 Average Final Accuracy: {avg_acc:.4f}")
    print(f"🎯 Average Macro F1: {avg_f1:.4f}")
    print("=" * 50)

    # 8. Aggregate SHAP Analysis (using data points from all folds)
    if explain_method == "shap.TreeExplainer" and all_shap_values_list:
        print(f"\n📈 Detailed SHAP Analysis based on all 5-fold data points...")

        # Concatenate SHAP values and feature matrices from all folds
        total_shap_values = np.vstack(all_shap_values_list)
        total_test_features = np.vstack(all_test_features_list)

        # Print detailed feature importance ranking (Class 1)
        print("\n📊 SHAP Top Feature Importance (Class 1):")
        feature_rank_dict = analyze_shap_feature_statistics(
            total_shap_values, total_test_features, final_all_feature_names
        )

        # Group SHAP Analysis
        abs_mean_values = analyze_shap_feature_statistics_by_group(
            total_shap_values, total_test_features, final_all_feature_names, feature_rank_dict
        )

        summarize_shap_by_group(abs_mean_values, final_all_feature_names)

        # Plot Summary Plot
        plt.figure()
        shap.summary_plot(total_shap_values, total_test_features, feature_names=final_all_feature_names, show=False)
        plt.savefig("./explain_output/shap_summary_plot_5fold.png", bbox_inches="tight")
        print("✅ SHAP summary plot saved to ./explain_output/shap_summary_plot_5fold.png")
        plt.close()


if __name__ == "__main__":

    Is_AI_Specific = False

    enabled_config = {
        "repo": True,
        "modification": True,
        "textual": True,
        "file": True,
        "action": True,
        "topic": True,
        "tfidf": False,
    }

    feature_df = pd.read_csv("_selected_reviews_with_features/reviews_contains_valid_by_LLM_lda_topics_selected.csv")
    label_df = pd.read_csv("labels/all_reviews_with_LLM_labels.csv")

    # Merge data.
    df = pd.merge(feature_df, label_df[["Comment_ID", "Change_Label"]], on="Comment_ID", how="inner")

    # Filter out human data (Human review rows)
    if Is_AI_Specific and "Source" in df.columns:
        print(f"Original row count: {len(df)}")
        df = df[df["Source"] != "Human"]
        print(f"Row count after removing Human source: {len(df)}")

        # ========================================================================================================
        # Verify if Trigger Mode and Model Configuration are suitable for binary encoding
        # ========================================================================================================

        # --- Verify Trigger Mode ---
        trigger_unique = df["Trigger_Mode"].dropna().unique()
        print(f"\n🔍 Verify Trigger_Mode unique values (after filtering Human): {trigger_unique}")
        trigger_ok = False
        if set(trigger_unique) <= {"auto", "manual"}:
            print("✅ Trigger_Mode only contains auto/manual, suitable for binary feature")
            trigger_ok = True
        else:
            print(f"⚠️ Trigger_Mode contains extra values: {set(trigger_unique) - {'auto', 'manual'} }, binary feature is not recommended")

        # --- Verify Model Configuration ---
        model_mapped = df["Model_Configured"].apply(map_model)
        model_unique = model_mapped.unique()
        print(f"🔍 Verify Model_Configured mapped unique values (after filtering Human): {model_unique}")
        # After filtering Human, NA and unknown are usually dropped, leaving only gpt-3.5 and gpt-4
        model_values_after_drop = set(model_unique) - {"NA", "unknown"}

        model_ok = False
        if model_values_after_drop <= {"gpt-3.5", "gpt-4"}:
            print("✅ Model_Configured (after removing NA/unknown) only contains gpt-3.5/gpt-4, suitable for binary feature")
            model_ok = True
        else:
            print(
                f"⚠️ Model_Configured contains extra values: {model_values_after_drop - {'gpt-3.5', 'gpt-4'} }, binary feature is not recommended")

        USE_BINARY_FEATURES = trigger_ok and model_ok

        if USE_BINARY_FEATURES:
            print("\n🔧 Verification passed: Automatically pass the instruction to the feature builder to use Binary encoding (action_use_binary=True)")
            enabled_config["action_use_binary"] = True
        else:
            print("\n🔧 Verification failed or data contains extra states: Fallback to keep using one-hot encoding (action_use_binary=False)")
            enabled_config["action_use_binary"] = False

    # Perform training with 5-fold CV.
    train_rf_with_tfidf_and_structured_5fold(
        df=df,
        enabled_feature_types=enabled_config,
        text_column="Cleaned_Body",
        label_column="Change_Label",
        explain_method="shap.TreeExplainer",
    )
