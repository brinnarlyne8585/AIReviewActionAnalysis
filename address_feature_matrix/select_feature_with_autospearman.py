# run_autospearman_feature_selection.py

import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, StrVector
import rpy2.robjects as robjects
import os

# Enable automatic conversion between pandas and R dataframes.
pandas2ri.activate()

# Load the R package containing AutoSpearman.
Rnalytica = importr('Rnalytica')

def run_autospearman_feature_selection(input_path: str, output_path: str) -> list:
    """
    Read data from a CSV file, automatically identify numerical features, use AutoSpearman to remove highly correlated features, and save the selected feature set to a new CSV file.

    :param input_path: Path to the input CSV file
    :param output_path: Path to the output CSV file containing the selected features
    :return: List of retained feature names
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file does not exist: {input_path}")

    print(f"📥 Reading file: {input_path}")
    data = pd.read_csv(input_path)

    # Automatically identify columns unsuitable for analysis (text, boolean, or unique value columns).
    excluded_columns_auto = []
    col_types = data.dtypes.to_dict()
    for col in data.columns:
        dtype = col_types[col]
        if dtype == 'object' or dtype == 'bool' or data[col].nunique() <= 1:
            excluded_columns_auto.append(col)

    # Select numeric columns and exclude unsuitable ones.
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    feature_names = [f for f in numeric_features if f not in excluded_columns_auto]

    excluded_prefixes = ['LDA_Topic_Prob_','LDA_Topic_Label']
    feature_names = [
        f for f in feature_names
        if not any(f.startswith(prefix) for prefix in excluded_prefixes)
    ]

    print(f"🔍 Automatically detected {len(feature_names)} numerical features, running AutoSpearman for decorrelation...")
    print(f"{feature_names}")

    # Construct input for AutoSpearman
    X = data[feature_names]

    # Call AutoSpearman
    results = Rnalytica.AutoSpearman(dataset=X, metrics=StrVector(feature_names))

    # Convert to Python list
    selected_features = list(results)

    # ✅ Print retained & removed features
    print("✅ Features retained by AutoSpearman:")
    print(selected_features)

    removed_features = sorted(set(feature_names) - set(selected_features))
    if removed_features:
        print("❌ Features removed by AutoSpearman:")
        print(removed_features)
    else:
        print("🎉 No features were removed by AutoSpearman")

    selected_features = list(results)  # ✅ Correctly extract the list of selected feature names
    filtered_numeric = X[selected_features]

    # Add back non-numeric columns (not passed into AutoSpearman)
    other_columns = [col for col in data.columns if col not in X.columns]
    other_data = data[other_columns]

    # Merge final feature data
    filtered_data = pd.concat([filtered_numeric, other_data], axis=1)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_data.to_csv(output_path, index=False)

    print("✅ AutoSpearman feature selection completed!")
    print(f"👉 Retained numeric feature columns (total {len(selected_features)}):")
    print(selected_features)
    print(f"📁 File written to: {output_path}")

    return selected_features


if __name__ == "__main__":

    input_csv = "_selected_reviews_with_features/reviews_contains_valid_by_LLM_lda_topics.csv"
    output_csv = input_csv.replace(".csv","_selected.csv")
    run_autospearman_feature_selection(input_csv, output_csv)
