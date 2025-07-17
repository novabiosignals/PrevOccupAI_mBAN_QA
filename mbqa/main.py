# main libraries
from typing import List, Dict
import os
import pandas as pd
from tqdm import tqdm

# internal libraries
from .loader import discover_dataset_files, load_acc_file, load_subject_mac_map, select_random_sample
from .quality_assessment import assess_acc_quality
from .viz import export_session_report_to_pdf
from .constants import BASE_DIR, SUBJECTS_CSV_PATH, ACC_COLUMNS

def get_session_from_path(path: str) -> str:
    """
    Extract session ID from the file path.

    :param path: Path to the accelerometer file.
    :return: Session ID in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    # Example: opensignals_84FD27E506E8_2024-05-27_10-39-20.txt to 2024-05-27_10-39-20
    return "_".join(os.path.basename(path).split("_")[2:]).replace(".txt", "")

def quality_exec(
    use_sample: bool = False, 
    n_files: int = 40, 
    generate_reports: bool = True
):
    """
    Process all sessions in the dataset, assess their quality,
    and save results to CSV and optionally to PDF/plots.
    
    :param use_sample: Whether to use a representative random sample for QA.
    :type use_sample: bool
    :param n_files: Number of files to sample if use_sample is True.
    :type n_files: int
    :param generate_reports: Whether to generate plots and PDF reports.
    :type generate_reports: bool
    """
    # Ensure the reports directory exists for saving figures
    os.makedirs("reports", exist_ok=True)

    results = []
    print("Loading subject MAC mapping...")

    if not os.path.exists(SUBJECTS_CSV_PATH):
        raise FileNotFoundError(f"Subjects CSV file not found: {SUBJECTS_CSV_PATH}")
    
    device_to_subject, subject_mac_map = load_subject_mac_map(SUBJECTS_CSV_PATH)

    # --- NEW: Discover all records first
    recs_iter = discover_dataset_files(device_to_subject, subject_mac_map, base_dir=BASE_DIR)
    all_recs = list(recs_iter) if recs_iter is not None else []

    # Optionally select a sample
    if use_sample:
        print(f"Sampling {n_files} files for quality assessment.")
        recs = select_random_sample(
            all_recs, n_files=n_files, min_n_subjects=5, min_n_groups=2
        )
        print(f"Selected {len(recs)} representative records for sampling.")
    else:
        recs = all_recs
        print(f"Using ALL records: {len(recs)} files.")

    if not recs:
        print("No valid records found for sampling. Exiting.")
        return

    for rec in tqdm(recs, desc="Sessions"):
        session_id_left = get_session_from_path(rec["left"]) if rec.get("left") else None
        session_id_right = get_session_from_path(rec["right"]) if rec.get("right") else None

        processed_sides = []
        session_ids = {}

        # Process LEFT side if available and valid
        if rec.get("left"):
            try:
                df_left = load_acc_file(rec["left"])
                if not df_left.empty:
                    df_left['group'] = rec["group"]
                    df_left['subject'] = rec["subject"]
                    df_left['day'] = rec["day"]
                    df_left['session'] = session_id_left
                    df_left['side'] = "left"

                    qa_left = assess_acc_quality(df_left, generate_reports=generate_reports)
                    for axis in ACC_COLUMNS:
                        results.append({
                            "group": rec["group"],
                            "subject": rec["subject"],
                            "day": rec["day"],
                            "session": session_id_left,
                            "side": "left",
                            "axis": axis,
                            "file": rec["left"],
                            **qa_left[axis],
                        })
                    processed_sides.append("left")
                    session_ids["left"] = session_id_left
                else:
                    print(f"Empty data for LEFT side in session {rec['session']}.")
            except Exception as e:
                print(f"Error loading LEFT data for session {rec['session']}: {e}")

        # Process RIGHT side if available and valid
        if rec.get("right"):
            try:
                df_right = load_acc_file(rec["right"])
                if not df_right.empty:
                    df_right['group'] = rec["group"]
                    df_right['subject'] = rec["subject"]
                    df_right['day'] = rec["day"]
                    df_right['session'] = session_id_right
                    df_right['side'] = "right"

                    qa_right = assess_acc_quality(df_right, generate_reports=generate_reports)
                    for axis in ACC_COLUMNS:
                        results.append({
                            "group": rec["group"],
                            "subject": rec["subject"],
                            "day": rec["day"],
                            "session": session_id_right,
                            "side": "right",
                            "axis": axis,
                            "file": rec["right"],
                            **qa_right[axis],
                        })
                    processed_sides.append("right")
                    session_ids["right"] = session_id_right
                else:
                    print(f"Empty data for RIGHT side in session {rec['session']}.")
            except Exception as e:
                print(f"Error loading RIGHT data for session {rec['session']}: {e}")

        # Generate report if at least one side is processed
        if generate_reports and processed_sides:
            # Use left session id if available, else right, for prefix
            prefix_session_id = session_ids.get("left") or session_ids.get("right")
            session_prefix = f"{rec['group']}_subject{rec['subject']}_session{prefix_session_id}"
            export_session_report_to_pdf(
                outdir="reports",
                session_prefix=session_prefix,
                pdf_name=f"reports/{session_prefix}_report.pdf"
            )

    df = pd.DataFrame(results)
    df.to_csv("ACC_quality_summary.csv", index=False)
    print("✓ Summary saved  →  ACC_quality_summary.csv")


if __name__ == "__main__":
    # To run on a representative sample (faster, for debugging/QA), use:
    quality_exec(use_sample=True, n_files=40, generate_reports=True)

    # To process the entire dataset, use:
    # quality_exec(use_sample=False, generate_reports=True)