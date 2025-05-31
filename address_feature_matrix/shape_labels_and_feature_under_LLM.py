from config import BASE_DIR
import pandas as pd
import os

def generate_llm_labels_from_resolution():
    input_file = f"{BASE_DIR}/llm_analysis/output/reviews(llm_input)(consider_path)/Addressed_openai-o3-mini_p=4.7(1)_based_Suggestion_openai-o3-mini_p=3.12(1)(f).csv"
    output_dir = f"{BASE_DIR}/address_feature_matrix/labels"
    output_file = os.path.join(output_dir, "all_reviews_with_LLM_labels.csv")

    # Create the output directory (if it doesn't exist).
    os.makedirs(output_dir, exist_ok=True)

    # Load the data.
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    # Only Remain 2,0  2,1  2,2.
    df = df[
        df["Resolution_Formated"].astype(str).str.startswith("2") &
        (df["Resolution_Formated"].astype(str) != "2,-1")
        ]

    label_mapping = {
        "2,0": 0,
        "2,1": 1,
        "2,2": 1
    }

    # Apply the mapping and process the output columns.
    df = df[["Comment_URL", "Resolution_Formated"]].copy()
    df["Change_Label"] = df["Resolution_Formated"].map(label_mapping)
    df = df.dropna(subset=["Change_Label"])
    df = df.rename(columns={"Comment_URL": "Comment_ID"})

    # Output file.
    df[["Comment_ID", "Change_Label"]].to_csv(output_file, index=False)
    print(f"✅ LLM labels written to: {output_file} (total: {len(df)} entries)")



def filter_feature_rows_with_valid_llm_labels():
    # Input path.
    feature_file = f"{BASE_DIR}/address_feature_matrix/_all_review_with_features/all_reviews_with_features.csv"
    llm_label_file = f"{BASE_DIR}/address_feature_matrix/labels/all_reviews_with_LLM_labels.csv"
    output_file = f"{BASE_DIR}/address_feature_matrix/_selected_reviews_with_features/reviews_contains_valid_by_LLM.csv"

    # Load data
    feature_df = pd.read_csv(feature_file)
    llm_label_df = pd.read_csv(llm_label_file)

    # Clean field names (to avoid issues like spaces).
    feature_df.columns = feature_df.columns.str.strip()
    llm_label_df.columns = llm_label_df.columns.str.strip()

    # Filter: Keep only the entries whose Comment_ID exists in the LLM labels.
    filtered_df = feature_df[feature_df["Comment_ID"].isin(llm_label_df["Comment_ID"])].copy()

    # Save results.
    filtered_df.to_csv(output_file, index=False)
    print(f"✅ Filtered features saved to: {output_file} (total: {len(filtered_df)} entries)")


if __name__ == "__main__":

    # Generate labels based on the results from the large language model:
    generate_llm_labels_from_resolution()

    # Reshape the feature file based on the results from the large language model:
    filter_feature_rows_with_valid_llm_labels()
