import os
import re
import xlsxwriter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score

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
        print(f"| {'-' * 18} | {'-' * 6} |")
        print(f"| Overall Accuracy | {metrics_dict['overall_accuracy']:.4f} |")
        print(f"| Micro-F1         | {metrics_dict['micro_f1']:.4f} |")
        print(f"| Macro-F1         | {metrics_dict['macro_f1']:.4f} |")
    else:
        print(
            f"{name}\t{metrics_dict['overall_accuracy']:.4f}\t{metrics_dict['micro_f1']:.4f}\t{metrics_dict['macro_f1']:.4f}")


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
        "macro_f1": (f1_score(y_true_stage1, y_pred_stage1, average='macro')),
        "kappa": cohen_kappa_score(y_true_stage1, y_pred_stage1),
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

    kappa = cohen_kappa_score(y_true_mapped, y_pred_mapped)
    if kappa < 0.7:
        print(y_true_mapped)
        print(y_pred_mapped)

    return {
        "overall_accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": (f1_score(y_true_mapped, y_pred_mapped, average='macro')),
        "kappa": cohen_kappa_score(y_true_mapped, y_pred_mapped),
    }


def evaluate_coarse_grained(y_true, y_pred, is_process_friendly):
    """Coarse-grained classification evaluation."""
    # Label mapping.
    coarse_map = {
        "0": "not_addressed", "1": "not_addressed",
        "2,-1": "not_addressed", "2,0": "not_addressed",
        "2,1": "addressed", "2,2": "addressed",
    }
    y_true_c = y_true.map(coarse_map)
    y_pred_c = y_pred.map(coarse_map)

    # Core metrics calculation.
    accuracy = accuracy_score(y_true_c, y_pred_c)
    # Binary F1 usually focuses on positive class, but macro/micro avg still applies
    # Let's keep micro/macro as before for consistency in printing
    micro_f1 = f1_score(y_true_c, y_pred_c, average='micro')
    macro_f1 = f1_score(y_true_c, y_pred_c, average='macro')

    # Print detailed process information.
    if is_process_friendly:
        print("\n🧮 Confusion Matrix:")
        labels = ["not_addressed", "addressed"]
        cm = confusion_matrix(y_true_c, y_pred_c, labels=labels)
        print_excel_friendly_confusion_matrix(cm, [f"T:{l}" for l in labels], [f"P:{l}" for l in labels])

    return {
        "overall_accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "kappa": cohen_kappa_score(y_true_c, y_pred_c),
    }


def evaluate_prediction_with_file(auto_file: str, labeled_file: str, task_name: str,
                                  do_fine_grained=True,
                                  do_coarse_grained=True,
                                  do_stage1=True,
                                  do_stage2=True,
                                  is_process_friendly=True, ):
    auto_df = pd.read_csv(auto_file)
    labeled_df = pd.read_csv(labeled_file)

    return evaluate_prediction_with_df(auto_df, labeled_df,
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
        is_process_friendly=True, ):
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
            "macro_f1": f1_score(y_true, y_pred, average='macro'),
            "kappa": cohen_kappa_score(y_true, y_pred),
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
        print_core_metrics(name, values,
                           is_process_friendly=is_process_friendly)  # Modify the parameter passing method.

    return {
        "Fine-Grained": fine_metrics,
        "Coarse-Grained": coarse_metrics,
        "Stage1": stage1_metrics,
        "Stage2": stage2_metrics if stage2_metrics else {
            "overall_accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0, "kappa": 0.0
        }
    }


LLM_MODEL_COMBOS = [
    # ("openai-gpt-4.1", "openai-gpt-4.1"),
    # ("openai-gpt-4o", "openai-gpt-4o"),
    # ("openai-o4-mini", "openai-o4-mini"),
    # ("openai-o3-mini", "openai-o3-mini"),
    # ("claude-3-7-sonnet", "claude-3-7-sonnet"),
    # ("claude-3-5-haiku", "claude-3-5-haiku"),
    # ("deepseek-r1", "deepseek-r1"),
    # ("deepseek-v3", "deepseek-v3"),

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
RUNS = [1, 2, 3, 4, 5]

STAGES = ["Fine-Grained", "Coarse-Grained", "Stage1", "Stage2"]
METRICS = ["overall_accuracy", "micro_f1", "macro_f1", "kappa"]
STAGE_MAP = {
    "Fine-Grained": "Fine",
    "Coarse-Grained": "Coarse",
    "Stage1": "Stage1",
    "Stage2": "Stage2"
}
METRIC_LABEL = {
    "overall_accuracy": "Overall",
    "micro_f1": "Micro",
    "macro_f1": "Macro",
    "kappa": "Kappa",
}


def evaluate_all_models_to_excel(input_dir="output", labeled_dir="labeled",
                                 output_file="llm_evaluation_blocked(stage-1).xlsx"):
    workbook = xlsxwriter.Workbook(output_file)

    # 🟢 Create Summary Sheet First
    summary_sheet = workbook.add_worksheet("Summary")
    bold_center = workbook.add_format({'bold': True, 'align': 'center'})
    center = workbook.add_format({'align': 'center'})
    pct_format = workbook.add_format({'num_format': '0.0%', 'align': 'center'})

    # Headers for Summary
    # Added "Average of Three" as requested
    summary_categories = ["Average of Three", "🔴 Total 150 Review Comments", "🔵 Human Review Comment",
                          "🟠 Patch-Level Review Comment", "🟢 File-Level Review Comment"]
    # Map friendly names to internal keys for logic
    summary_cat_keys = ["avg_of_3", "total", "human", "patch_level", "file_level"]

    # Added "Coarse" to stages
    summary_stages = ["Fine", "Coarse", "Stage1", "Stage2"]
    summary_metrics = ["OverallAcc.", "Macro-F1"]

    # Write Summary Headers
    # Row 0: Category Headers
    summary_sheet.write(0, 0, "LLMs", bold_center)
    summary_sheet.write(0, 1, "5 runs", bold_center)

    col_ptr = 2
    for cat in summary_categories:
        # Each category has 4 stages * 2 metrics = 8 columns
        summary_sheet.merge_range(0, col_ptr, 0, col_ptr + 7, cat, bold_center)
        col_ptr += 8

    # Row 1: Stage Headers
    summary_sheet.write(1, 0, "", bold_center)
    summary_sheet.write(1, 1, "", bold_center)
    col_ptr = 2
    for _ in summary_categories:
        for stage in summary_stages:
            summary_sheet.merge_range(1, col_ptr, 1, col_ptr + 1, stage, bold_center)
            col_ptr += 2

    # Row 2: Metric Headers
    summary_sheet.write(2, 0, "", bold_center)
    summary_sheet.write(2, 1, "", bold_center)
    col_ptr = 2
    for _ in summary_categories:
        for _ in summary_stages:
            for metric in summary_metrics:
                summary_sheet.write(2, col_ptr, metric, bold_center)
                col_ptr += 1

    summary_row_ptr = 3

    for model_combos in LLM_MODEL_COMBOS:
        task1_llm, task2_llm = model_combos
        combo_name = f"{task1_llm}+{task2_llm}"
        sheet_name_clean = combo_name.replace("openai-", "").replace("deepseek-", "").replace("claude-", "")
        if len(sheet_name_clean) > 31:
            sheet_name_clean = sheet_name_clean[:31]

        # Check if sheet already exists (case sensitive) to avoid crash?
        # But we create fresh workbook.
        # xlsxwriter allows 31 chars max for sheet name.

        worksheet = workbook.add_worksheet(sheet_name_clean)

        row_ptr = 0
        total150_per_run = {}

        # Store metrics for summary for this model combo
        # Structure: { 'human': { 'Fine': {'acc':.., 'f1':..}, ... }, 'total': ... }
        current_model_stats = {k: {} for k in summary_cat_keys if k != "avg_of_3"}

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
                    # print(result)
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
            # print(df)
            for r in range(df.shape[0]):
                for c in range(df.shape[1]):
                    worksheet.write(row_ptr + r, c, df.iloc[r, c], center)
            row_ptr += df.shape[0] + 2

            # Extract AVG for Summary (Review Type)
            for stage_key, stage_name in [("Fine", "Fine-Grained"), ("Coarse", "Coarse-Grained"), ("Stage1", "Stage1"),
                                          ("Stage2", "Stage2")]:
                acc_col = f"{STAGE_MAP[stage_name]}_Overall"
                f1_col = f"{STAGE_MAP[stage_name]}_Macro"

                try:
                    val_acc = df.iloc[-2][acc_col]
                    val_f1 = df.iloc[-2][f1_col]

                    if stage_key not in current_model_stats[review_type]:
                        current_model_stats[review_type][stage_key] = {}
                    current_model_stats[review_type][stage_key]['acc'] = val_acc
                    current_model_stats[review_type][stage_key]['f1'] = val_f1
                except KeyError:
                    pass

        # 🔴 Total 150 Evaluation
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

        # Extract AVG for Summary (Total)
        for stage_key, stage_name in [("Fine", "Fine-Grained"), ("Coarse", "Coarse-Grained"), ("Stage1", "Stage1"),
                                      ("Stage2", "Stage2")]:
            acc_col = f"{STAGE_MAP[stage_name]}_Overall"
            f1_col = f"{STAGE_MAP[stage_name]}_Macro"
            try:
                val_acc = df_total.iloc[-2][acc_col]
                val_f1 = df_total.iloc[-2][f1_col]

                if stage_key not in current_model_stats['total']:
                    current_model_stats['total'][stage_key] = {}
                current_model_stats['total'][stage_key]['acc'] = val_acc
                current_model_stats['total'][stage_key]['f1'] = val_f1
            except KeyError:
                pass

        # CALCULATE AVERAGE OF THREE
        current_model_stats['avg_of_3'] = {}
        for stage in summary_stages:
            current_model_stats['avg_of_3'][stage] = {}
            # Average Human, Patch, File
            human_acc = current_model_stats.get('human', {}).get(stage, {}).get('acc', 0)
            patch_acc = current_model_stats.get('patch_level', {}).get(stage, {}).get('acc', 0)
            file_acc = current_model_stats.get('file_level', {}).get(stage, {}).get('acc', 0)

            human_f1 = current_model_stats.get('human', {}).get(stage, {}).get('f1', 0)
            patch_f1 = current_model_stats.get('patch_level', {}).get(stage, {}).get('f1', 0)
            file_f1 = current_model_stats.get('file_level', {}).get(stage, {}).get('f1', 0)

            current_model_stats['avg_of_3'][stage]['acc'] = (human_acc + patch_acc + file_acc) / 3
            current_model_stats['avg_of_3'][stage]['f1'] = (human_f1 + patch_f1 + file_f1) / 3

        # WRITE TO SUMMARY SHEET FOR THIS MODEL
        summary_sheet.write(summary_row_ptr, 0, combo_name, center)
        summary_sheet.write(summary_row_ptr, 1, "Avg", center)

        col_ptr = 2

        # Order: AvgOf3 -> Total -> Human -> Patch -> File
        for cat in summary_cat_keys:
            for stage in summary_stages:
                acc_val = current_model_stats.get(cat, {}).get(stage, {}).get('acc', 0)
                f1_val = current_model_stats.get(cat, {}).get(stage, {}).get('f1', 0)

                summary_sheet.write(summary_row_ptr, col_ptr, acc_val, pct_format)
                summary_sheet.write(summary_row_ptr, col_ptr + 1, f1_val, pct_format)
                col_ptr += 2

        summary_row_ptr += 1

    workbook.close()
    print(f"\n✅ Save as: {output_file}")


if __name__ == "__main__":
    evaluate_all_models_to_excel()