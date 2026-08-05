import pandas as pd
import os

def compare_submissions(file_best, file_new):
    if not os.path.exists(file_best):
        print(f"Error: {file_best} not found.")
        return
    if not os.path.exists(file_new):
        print(f"Error: {file_new} not found.")
        return

    df_best = pd.read_csv(file_best)
    df_new = pd.read_csv(file_new)
    # Fill NaN to avoid NaN != NaN issues
    df_best = df_best.fillna('')
    df_new = df_new.fillna('')

    if len(df_best) != len(df_new):
        print(f"Warning: Files have different lengths! Best: {len(df_best)}, New: {len(df_new)}")
    
    # Merge on id
    merged = pd.merge(df_best, df_new, on='id', suffixes=('_best', '_new'))
    
    total = len(merged)
    
    # Check differences in has_anomaly
    diff_anomaly = merged[merged['has_anomaly_best'] != merged['has_anomaly_new']]
    print(f"\nTotal rows: {total}")
    print(f"Rows with different 'has_anomaly': {len(diff_anomaly)} ({len(diff_anomaly)/total:.2%})")
    
    # Check differences in primary_anomaly_type
    diff_type = merged[merged['primary_anomaly_type_best'] != merged['primary_anomaly_type_new']]
    print(f"Rows with different 'primary_anomaly_type': {len(diff_type)} ({len(diff_type)/total:.2%})")
    
    # Check differences in start/end idx
    diff_start = merged[merged['primary_start_idx_best'] != merged['primary_start_idx_new']]
    diff_end = merged[merged['primary_end_idx_best'] != merged['primary_end_idx_new']]
    print(f"Rows with different 'primary_start_idx': {len(diff_start)} ({len(diff_start)/total:.2%})")
    print(f"Rows with different 'primary_end_idx': {len(diff_end)} ({len(diff_end)/total:.2%})")

    # Overall differences (any column except id)
    cols_to_compare = [c for c in df_best.columns if c != 'id']
    is_diff = pd.Series([False] * len(merged))
    for col in cols_to_compare:
        is_diff |= (merged[f"{col}_best"] != merged[f"{col}_new"])
    
    diff_all = merged[is_diff]
    print(f"Rows with ANY difference: {len(diff_all)} ({len(diff_all)/total:.2%})")

    if len(diff_all) > 0:
        print("\nExamples of differences (First 10):")
        # Select some columns to show
        show_cols = ['id', 'has_anomaly_best', 'has_anomaly_new', 'primary_anomaly_type_best', 'primary_anomaly_type_new', 'all_spans_best', 'all_spans_new']
        # Set display options for better visibility
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(diff_all[show_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    best_path = r"提交结果\submission_best.csv"
    new_path = r"提交结果\submission.csv"
    compare_submissions(best_path, new_path)
