import numpy as np
import pandas as pd
from address_feature_matrix.extract_topic_with_LDA import TOPIC_NUMBER

FEATURE_GROUP_MAPPING = {
    "repo": [
        'Repo_Total_File_Number',
        'Repo_Total_File_Size',
        'Repo_Total_PR_Number',
        'Repo_Total_Issue_Number',
        'Repo_Total_NonBot_Contributor_Number',
    ],
    "modification": [
        'Author_Is_Anonymous', 'Author_Is_Bot', 'Author_Past_Commit_Count',

        'Commit_Accumulated_Changed_File_Count',

        'Commit_Accumulated_Total_Change_Line_Count',
        'Commit_Accumulated_Total_Add_Line_Count',
        'Commit_Accumulated_Total_Del_Line_Count',

        'Commit_Base_Total_Line_Count',

        'Is_Programming_File',
        'File_Depth',

        'File_Change_Line_Count',
        'File_Add_Line_Count',
        'File_Del_Line_Count',

        'File_Base_Line_Count',

        'Comment_Adds',
        'Comment_Dels',
        'Comment_Changes',
    ],
    "textual": [
        'Text_Length',
        'Has_Inline_Code',
        'Has_Multiline_Code',
        'Code_Total_Length',
        'Code_Text_Ratio',

        *[f"LDA_Topic_Prob_{i}" for i in range(TOPIC_NUMBER)],

        "Timeline_Index",
        "Cumulative_Prior_Text_Length",
        "Thread_Companion_Count",
    ],
    "source": [
        'Is_Human',
        'Is_File_Level_Action',
        'Source_aidar-freeed/ai-codereviewer',
        'Source_anc95/ChatGPT-CodeReview',
        'Source_coderabbitai/ai-pr-reviewer',
        'Source_mattzcarey/code-review-gpt',
        'Trigger_auto',
        'Trigger_manual',
        'Model_gpt-3.5',
        'Model_gpt-4',
    ]
}

# ------------------------------ Add topic feature construction. ------------------------------ #
def build_topic_feature(df: pd.DataFrame) -> pd.DataFrame:
    topic_prob_columns = [col for col in df.columns if col.startswith("LDA_Topic_Prob_")]
    return df[topic_prob_columns]


# -------------------------------- Descriptive features for the repo.  -------------------------------- #
def build_repo_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Repo_Total_File_Number",
        "Repo_Total_File_Size",
        "Repo_Total_PR_Number",
        "Repo_Total_Issue_Number",
        "Repo_Total_NonBot_Contributor_Number",
    ]
    # Actually existing columns.
    available_columns = [col for col in columns if col in df.columns]
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        print(f"⚠️ The following columns are missing in the input data and will be ignored:{missing_columns}")

    result_df = df[available_columns].copy()

    # Convert boolean values to integers (True → 1, False → 0).
    if "Repo_Has_Custom_Config" in result_df.columns:
        result_df["Repo_Has_Custom_Config"] = result_df["Repo_Has_Custom_Config"].astype(int)

    return result_df


# -------------------------------- Descriptive features for the modification. -------------------------------- #
def build_modification_features(df: pd.DataFrame) -> pd.DataFrame:

    # Convert boolean values to integers (1/0) and rename the fields with the 'Author_' prefix.
    author_type_encoded = df[["Is_Anonymous", "Is_Bot", "Past_Commit_Count"]].copy()
    author_type_encoded = author_type_encoded.rename(columns={
        "Is_Anonymous": "Author_Is_Anonymous",
        "Is_Bot": "Author_Is_Bot",
        "Past_Commit_Count": "Author_Past_Commit_Count"
    })
    author_type_encoded["Author_Is_Anonymous"] = author_type_encoded["Author_Is_Anonymous"].astype(int)
    author_type_encoded["Author_Is_Bot"] = author_type_encoded["Author_Is_Bot"].astype(int)
    author_type_encoded["Author_Past_Commit_Count"] = author_type_encoded["Author_Past_Commit_Count"].fillna(0).astype(int)

    numerical_columns = [
        "Commit_Accumulated_Changed_File_Count",
        "Commit_Accumulated_Total_Change_Line_Count",
        "Commit_Accumulated_Total_Add_Line_Count",
        "Commit_Accumulated_Total_Del_Line_Count",
        "Commit_Base_Total_Line_Count",
        "File_Change_Line_Count",
        "File_Add_Line_Count",
        "File_Del_Line_Count",
        "File_Base_Line_Count",
        "Comment_Adds",
        "Comment_Dels",
        "Comment_Changes",

        "Timeline_Index",
        "Cumulative_Prior_Text_Length",

        "Thread_Companion_Count",
    ]
    # Filter out the columns that actually exist.
    available_columns = [col for col in numerical_columns if col in df.columns]
    missing_columns = [col for col in numerical_columns if col not in df.columns]

    if missing_columns:
        print(f"⚠️ The following numerical columns are missing in the input data and will be ignored: {missing_columns}")

    return pd.concat([author_type_encoded, df[available_columns].copy()], axis=1)

# -------------------------------- Textual descriptive features for the comment.  -------------------------------- #
def build_textual_features(df: pd.DataFrame) -> pd.DataFrame:
    textual_columns = [
        "Text_Length",
        "Has_Inline_Code",
        "Has_Multiline_Code",
        "Code_Total_Length",
        "Code_Text_Ratio",
    ]

    available_columns = [col for col in textual_columns if col in df.columns]
    missing_columns = [col for col in textual_columns if col not in df.columns]

    if missing_columns:
        print(f"⚠️ The following textual columns are missing in the input data and will be ignored:{missing_columns}")

    return df[available_columns].copy()


# -------------------------------- For file type features.  -------------------------------- #
def build_file_features(df: pd.DataFrame) -> pd.DataFrame:
    # Encode whether the type is 'programming'.
    is_programming = df["File_Final_Type"] == "programming"
    file_depth = df["File_Depth"]
    return pd.DataFrame({
        "Is_Programming_File": is_programming.astype(int),
        "File_Depth": file_depth.astype(int),
    })

def encode_trigger_mode(trigger_str):
    if trigger_str == "human":
        return 2
    elif trigger_str == "auto":
        return 0
    elif trigger_str == "manual":
        return 1
    else:
        return -1

def build_action_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Trigger mode one-hot encoding.
    trigger_encoded = pd.get_dummies(df["Trigger_Mode"].fillna("NA"), prefix="Trigger")
    # Remove the Trigger_human column (if it exists).
    if "Trigger_human" in trigger_encoded.columns:
        trigger_encoded = trigger_encoded.drop(columns=["Trigger_human"])

    # 2. Model configuration one-hot encoding (normalized to gpt-3.5, gpt-4, unknown, NA).
    def map_model(model):
        if isinstance(model, str):
            if model.startswith("gpt-3.5"):
                return "gpt-3.5"
            elif model.startswith("gpt-4"):
                return "gpt-4"
            elif model == "unknown":
                return "unknown"
        return "NA"

    model_group = df["Model_Configured"].apply(map_model)
    model_encoded = pd.get_dummies(model_group, prefix="Model")
    # Remove NA (overlaps with Is_Human).
    if "Model_NA" in model_encoded.columns:
        model_encoded = model_encoded.drop(columns=["Model_NA"])
    # Remove NA (overlaps with Is_Human).
    if "Model_unknown" in model_encoded.columns:
        model_encoded = model_encoded.drop(columns=["Model_unknown"])

    # 3. Source structure (whether it is human/action/file/patch).
    df["Is_Human"] = (df["Source"] == "Human").astype(int)
    df["Is_File_Level_Action"] = df["Source"].isin([
        "anc95/ChatGPT-CodeReview", "mattzcarey/code-review-gpt"
    ]).astype(int)

    # 4. Action source one-hot encoding (optional, can be commented out).
    action_name_encoded = pd.get_dummies(df["Source"], prefix="Source")
    # Remove information already represented by Is_Human (to avoid redundancy).
    if "Source_Human" in action_name_encoded.columns:
        action_name_encoded = action_name_encoded.drop(columns=["Source_Human"])

    return pd.concat([
        trigger_encoded,
        model_encoded,
        df[["Is_Human", "Is_File_Level_Action"]],
        action_name_encoded
    ], axis=1)


def assemble_feature_matrix(df: pd.DataFrame, enabled_feature_types: dict) -> pd.DataFrame:
    feature_parts = []

    if enabled_feature_types.get("repo"):
        feature_parts.append(build_repo_features(df))
    if enabled_feature_types.get("modification"):
        feature_parts.append(build_modification_features(df))
    if enabled_feature_types.get("textual"):
        feature_parts.append(build_textual_features(df))
    if enabled_feature_types.get("file"):
        feature_parts.append(build_file_features(df))
    if enabled_feature_types.get("action"):
        feature_parts.append(build_action_features(df))
    if enabled_feature_types.get("topic"):
        feature_parts.append(build_topic_feature(df))

    full_feature_matrix = pd.concat(feature_parts, axis=1)
    return full_feature_matrix

