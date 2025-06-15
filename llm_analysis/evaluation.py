import os
import re
import xlsxwriter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

VALID_LABELS = ["0", "1", "2,-1", "2,0", "2,1", "2,2"]


def clean_comment_url(url):
    return re.sub(r"^[^h]*", "", str(url).strip()).replace(" ", "")


def print_excel_friendly_confusion_matrix(cm, row_labels, col_labels):
    """Print the confusion matrix in a format suitable for copying into Excel."""
    header = '\t' + '\t'.join(col_labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = row_labels[i] + '\t' + '\t'.join(map(str, row))
        print(row_str)


def print_core_metrics(name, metrics_dict, is_process_friendly):
    """Standardize the core metrics printing function."""
    if is_process_friendly:
        print(f"\n📌 {name} Detailed Report")
        print(f"| {'Metric':<18} | {'Value':<6} |")
        print(f"| {'-'*18} | {'-'*6} |")
        print(f"| Overall Accuracy | {metrics_dict['overall_accuracy']:.4f} |")
        print(f"| Micro-F1         | {metrics_dict['micro_f1']:.4f} |")
        print(f"| Macro-F1         | {metrics_dict['macro_f1']:.4f} |")
    else:
        print(f"{name}\t{metrics_dict['overall_accuracy']:.4f}\t{metrics_dict['micro_f1']:.4f}\t{metrics_dict['macro_f1']:.4f}")


def evaluate_stage1(y_true, y_pred, is_process_friendly):
    """Phase 1 Evaluation: Classification of Validity Issues."""
    # Label mapping and confusion matrix computation.
    y_true_stage1 = y_true.map(lambda x: "valid" if x.startswith("2") else "not_valid")
    y_pred_stage1 = y_pred.map(lambda x: "valid" if x.startswith("2") else "not_valid")

    labels = ["not_valid", "valid"]
    cm = confusion_matrix(y_true_stage1, y_pred_stage1, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    # Core metrics calculation.
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    total_tp = tp + tn
    total_fp = fp + fn
    total_fn = fn + fp

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) > 0 else 0

    # Print detailed process information.
    if is_process_friendly:
        print("\n🧮 Confusion Matrix:")
        print_excel_friendly_confusion_matrix(cm, [f"T:{l}" for l in labels], [f"P:{l}" for l in labels])

        # Detailed metrics for each class.
        print("\n📊 Class-wise Metrics:")
        print(f"NonValid: P={tn / (tn + fp):.2f}\tR={tn / (tn + fn):.2f}")
        print(f"Valid:    P={tp / (tp + fp):.2f}\tR={tp / (tp + fn):.2f}")

    return {
        "overall_accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": (f1_score(y_true_stage1, y_pred_stage1, average='macro'))
    }


def evaluate_stage2(y_true, y_pred, is_process_friendly):
    """Phase 2 Evaluation: Resolution Status Classification."""
    # Data preprocessing.
    valid_mask = y_true.str.startswith("2") & y_pred.str.startswith("2")
    if not valid_mask.any():
        print("⚠️ No valid samples for stage2 evaluation")
        return None

    # Label mapping and confusion matrix computation.
    label_mapper = lambda x: "addressed" if x in ["2,1", "2,2"] else "not_addressed"
    y_true_mapped = y_true[valid_mask].map(label_mapper)
    y_pred_mapped = y_pred[valid_mask].map(label_mapper)

    labels = ["not_addressed", "addressed"]
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    # Core metrics calculation.
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    total_tp = tp + tn
    total_fp = fp + fn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + (fn + fp)) if (total_tp + (fn + fp)) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) > 0 else 0

    # Print detailed process information.
    if is_process_friendly:
        print("\n🧮 Confusion Matrix:")
        print_excel_friendly_confusion_matrix(cm, [f"T:{l}" for l in labels], [f"P:{l}" for l in labels])

        print("\n📊 Class-wise Metrics:")
        print(f"Not Addressed: P={tn / (tn + fp):.2f}\tR={tn / (tn + fn):.2f}")
        print(f"Addressed:     P={tp / (tp + fp):.2f}\tR={tp / (tp + fn):.2f}")

    return {
        "overall_accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": (f1_score(y_true_mapped, y_pred_mapped, average='macro'))
    }


def evaluate_coarse_grained(y_true, y_pred, is_process_friendly):
    """Coarse-grained classification evaluation."""
    # Label mapping.
    coarse_map = {
        "0": "invalid", "1": "invalid",
        "2,-1": "valid_na", "2,0": "valid_na",
        "2,1": "valid_addressed", "2,2": "valid_addressed",
    }
    y_true_c = y_true.map(coarse_map)
    y_pred_c = y_pred.map(coarse_map)

    # Core metrics calculation.
    accuracy = accuracy_score(y_true_c, y_pred_c)
    micro_f1 = f1_score(y_true_c, y_pred_c, average='micro')
    macro_f1 = f1_score(y_true_c, y_pred_c, average='macro')

    # Print detailed process information.
    if is_process_friendly:
        print("\n🧮 Confusion Matrix:")
        labels = ["invalid", "valid_na", "valid_addressed"]
        cm = confusion_matrix(y_true_c, y_pred_c, labels=labels)
        print_excel_friendly_confusion_matrix(cm, [f"T:{l}" for l in labels], [f"P:{l}" for l in labels])

    return {
        "overall_accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }


def evaluate_prediction_with_file(auto_file: str, labeled_file: str, task_name: str,
                        do_fine_grained=True,
                        do_coarse_grained=True,
                        do_stage1=True,
                        do_stage2=True,
                        is_process_friendly=True,):
    auto_df = pd.read_csv(auto_file)
    labeled_df = pd.read_csv(labeled_file)

    return evaluate_prediction_with_df(auto_df,labeled_df,
                                task_name,
                                do_fine_grained,
                                do_coarse_grained,
                                do_stage1,
                                do_stage2,
                                is_process_friendly)

def evaluate_prediction_with_df(
                        auto_df, labeled_df,
                        task_name: str,
                        do_fine_grained=True,
                        do_coarse_grained=True,
                        do_stage1=True,
                        do_stage2=True,
                        is_process_friendly=True,):
    print(f"\n📊 Evaluating the task: {task_name}")

    auto_df.columns = auto_df.columns.str.strip()
    labeled_df.columns = labeled_df.columns.str.strip()
    auto_df["Comment_URL"] = auto_df["Comment_URL"].apply(clean_comment_url)
    labeled_df["Comment_URL"] = labeled_df["Comment_URL"].apply(clean_comment_url)

    auto_df = auto_df.rename(columns={"Resolution_Formated": "Pred_Label"})
    labeled_df = labeled_df.rename(columns={"Final Result": "True_Label"})

    merged = pd.merge(auto_df[["Comment_URL", "Pred_Label"]],
                      labeled_df[["Comment_URL", "True_Label"]],
                      on="Comment_URL", how="inner")

    merged = merged[merged["True_Label"].isin(VALID_LABELS)]
    merged = merged[merged["Pred_Label"].isin(VALID_LABELS)]

    y_true = merged["True_Label"]
    y_pred = merged["Pred_Label"]

    # Standardized metrics collection.
    metrics = []

    # Perform the evaluation and collect the results.
    if do_fine_grained:
        fine_metrics = {
            "overall_accuracy": accuracy_score(y_true, y_pred),
            "micro_f1": f1_score(y_true, y_pred, average='micro'),
            "macro_f1": f1_score(y_true, y_pred, average='macro')
        }

        if is_process_friendly:
            print("\n🧮 Fine-Grained Confusion Matrix:")
            # Define the order of fine-grained labels (6 categories).
            labels = ["0", "1", "2,-1", "2,0", "2,1", "2,2"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            print_excel_friendly_confusion_matrix(cm, [f"T:{l}" for l in labels], [f"P:{l}" for l in labels])

        print_core_metrics("Fine-Grained (6-class)", fine_metrics, is_process_friendly)

    if do_coarse_grained:
        coarse_metrics = evaluate_coarse_grained(y_true, y_pred, is_process_friendly)
        metrics.append(("Coarse-Grained", coarse_metrics))

    if do_stage1:
        stage1_metrics = evaluate_stage1(y_true, y_pred, is_process_friendly)
        metrics.append(("Stage1", stage1_metrics))

    if do_stage2:
        stage2_metrics = evaluate_stage2(y_true, y_pred, is_process_friendly)
        if stage2_metrics:
            metrics.append(("Stage2", stage2_metrics))

    # Standardize the output of core metrics.
    print(f"\n🔍 {task_name} Core Metrics:")
    for name, values in metrics:
        print_core_metrics(name, values, is_process_friendly=is_process_friendly)  # Modify the parameter passing method.

    return {
        "Fine-Grained": fine_metrics,
        "Coarse-Grained": coarse_metrics,
        "Stage1": stage1_metrics,
        "Stage2": stage2_metrics if stage2_metrics else {
            "overall_accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0
        }
    }


LLM_MODEL_COMBOS = [
    ("openai-gpt-4.1", "deepseek-r1"),
    ("openai-gpt-4.1", "openai-o3-mini"),
    ("openai-gpt-4.1", "openai-o4-mini"),
    ("openai-gpt-4.1", "openai-gpt-4o"),

    ("deepseek-v3", "deepseek-r1"),
    ("deepseek-v3", "openai-o3-mini"),
    ("deepseek-v3", "openai-o4-mini"),
    ("deepseek-v3", "openai-gpt-4o"),

    ("claude-3-7-sonnet", "deepseek-r1"),
    ("claude-3-7-sonnet", "openai-o3-mini"),
    ("claude-3-7-sonnet", "openai-o4-mini"),
    ("claude-3-7-sonnet", "openai-gpt-4o"),
]
REVIEW_TYPES = ["human", "patch_level", "file_level"]
REVIEW_TYPE_LABEL = {
    "human": "🔵 Human Review Comment",
    "patch_level": "🟠 Patch-Level Review Comment",
    "file_level": "🟢 File-Level Review Comment"
}
RUNS = [1,2,3,4,5]

STAGES = ["Fine-Grained", "Coarse-Grained", "Stage1", "Stage2"]
METRICS = ["overall_accuracy", "micro_f1", "macro_f1"]
STAGE_MAP = {
    "Fine-Grained": "Fine",
    "Coarse-Grained": "Coarse",
    "Stage1": "Stage1",
    "Stage2": "Stage2"
}
METRIC_LABEL = {
    "overall_accuracy": "Overall",
    "micro_f1": "Micro",
    "macro_f1": "Macro"
}
def evaluate_all_models_to_excel(input_dir="output", labeled_dir="labeled", output_file="llm_evaluation_blocked(example).xlsx"):
    workbook = xlsxwriter.Workbook(output_file)

    for model_combos in LLM_MODEL_COMBOS:
        task1_llm, task2_llm = model_combos
        sheet_name = task1_llm + "+" +task2_llm
        sheet_name = sheet_name.replace("openai-","")
        sheet_name = sheet_name.replace("deepseek-", "")
        sheet_name = sheet_name.replace("claude-", "")
        worksheet = workbook.add_worksheet(sheet_name)
        bold_center = workbook.add_format({'bold': True, 'align': 'center'})
        center = workbook.add_format({'align': 'center'})
        row_ptr = 0

        total150_per_run = {}

        for review_type in REVIEW_TYPES:
            n_cols = 1 + len(STAGES) * len(METRICS)
            worksheet.merge_range(row_ptr, 0, row_ptr, n_cols - 1, REVIEW_TYPE_LABEL[review_type], bold_center)
            row_ptr += 1

            header = ["Run"]
            for stage in STAGES:
                for metric in METRICS:
                    header.append(f"{STAGE_MAP[stage]}_{METRIC_LABEL[metric]}")
            for col_idx, val in enumerate(header):
                worksheet.write(row_ptr, col_idx, val, bold_center)
            row_ptr += 1

            run_data = []

            for run_id in RUNS:
                row = [run_id]
                base_name = f"sampled_{review_type}_review"
                labeled_file = os.path.join(labeled_dir, f"(resolved){base_name}.csv")
                addressed_file = os.path.join(
                    input_dir, base_name,
                    f"Addressed_{task2_llm}_p=4.7({run_id})_based_Suggestion_{task1_llm}_p=3.12({run_id})(f).csv"
                )

                try:
                    result = evaluate_prediction_with_file(
                        addressed_file,
                        labeled_file,
                        task_name=f"{model_combos}-{review_type}-Run{run_id}",
                        do_fine_grained=True,
                        do_coarse_grained=True,
                        do_stage1=True,
                        do_stage2=True,
                        is_process_friendly=True,
                    )
                    print(result)
                    for stage in STAGES:
                        for metric in METRICS:
                            row.append(result.get(stage, {}).get(metric, np.nan))
                except Exception as e:
                    print(f"❌ Error in {model_combos} {review_type} Run {run_id}: {e}")
                    row += [np.nan] * (len(header) - 1)

                run_data.append(row)

                # 👉 Gather Total150
                if run_id not in total150_per_run:
                    total150_per_run[run_id] = {"pred": [], "label": []}
                try:
                    pred_df = pd.read_csv(addressed_file)
                    label_df = pd.read_csv(labeled_file)
                    total150_per_run[run_id]["pred"].append(pred_df)
                    total150_per_run[run_id]["label"].append(label_df)
                except Exception as e:
                    print(f"❌ Total150 Read Error in {model_combos} {review_type} Run {run_id}: {e}")

            df = pd.DataFrame(run_data, columns=header)
            avg = ["Avg"] + [df[col].mean() for col in df.columns[1:]]
            std = ["Std"] + [df[col].std() for col in df.columns[1:]]
            df.loc[len(df.index)] = avg
            df.loc[len(df.index)] = std
            print(df)
            for r in range(df.shape[0]):
                for c in range(df.shape[1]):
                    worksheet.write(row_ptr + r, c, df.iloc[r, c], center)
            row_ptr += df.shape[0] + 2

        # 🔴 Total 150
        worksheet.merge_range(row_ptr, 0, row_ptr, n_cols - 1, "🔴 Total 150 Review Comments", bold_center)
        row_ptr += 1
        worksheet.write_row(row_ptr, 0, header, bold_center)
        row_ptr += 1

        total150_data = []

        for run_id in RUNS:
            row = [run_id]
            try:
                pred_all = pd.concat(total150_per_run[run_id]["pred"], ignore_index=True)
                label_all = pd.concat(total150_per_run[run_id]["label"], ignore_index=True)

                os.makedirs("temp", exist_ok=True)
                pred_all_path = f"temp/total150_pred_{task1_llm}_{task2_llm}_{run_id}.csv"
                label_all_path = f"temp/total150_label_{task1_llm}_{task2_llm}_{run_id}.csv"
                pred_all.to_csv(pred_all_path, index=False)
                label_all.to_csv(label_all_path, index=False)

                result = evaluate_prediction_with_file(
                    pred_all_path, label_all_path,
                    task_name=f"{model_combos}-total150-Run{run_id}",
                    do_fine_grained=True,
                    do_coarse_grained=True,
                    do_stage1=True,
                    do_stage2=True,
                    is_process_friendly=True,
                )

                for stage in STAGES:
                    for metric in METRICS:
                        row.append(result.get(stage, {}).get(metric, np.nan))
            except Exception as e:
                print(f"❌ Total150 Error in {model_combos} Run {run_id}: {e}")
                row += [np.nan] * (len(header) - 1)

            total150_data.append(row)

        df_total = pd.DataFrame(total150_data, columns=header)
        avg = ["Avg"] + [df_total[col].mean() for col in df_total.columns[1:]]
        std = ["Std"] + [df_total[col].std() for col in df_total.columns[1:]]
        df_total.loc[len(df_total.index)] = avg
        df_total.loc[len(df_total.index)] = std

        for r in range(df_total.shape[0]):
            for c in range(df_total.shape[1]):
                worksheet.write(row_ptr + r, c, df_total.iloc[r, c], center)
        row_ptr += df_total.shape[0] + 2

    workbook.close()
    print(f"\n✅ Save as: {output_file}")


if __name__ == "__main__":

    evaluate_all_models_to_excel()