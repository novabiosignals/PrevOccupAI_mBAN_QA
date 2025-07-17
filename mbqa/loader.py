# main libraries
import os, glob
import pandas as pd
from typing import Tuple, Dict
import re
import random
from collections import defaultdict

# internal imports
from .adc_a import adc_to_acceleration
from .constants import *

# -----------------------------------------------------------------
def load_subject_mac_map(
    csv_path=os.path.join(BASE_DIR, "subjects_info.csv")
) -> tuple[dict, dict]:
    """
    Load the subject MAC assignment file and return:
      1. A mapping from (group, device_num) → subject_id
      2. A mapping from (group, subject_id) → {'left': mBAN_left, 'right': mBAN_right}
    
    :param csv_path: Path to the CSV file containing subject MAC assignments.
    
    :returns:
        device_to_subject (dict): Maps (group, device_num) → subject_id.
        subject_mac_map (dict): Maps (group, subject_id) → {'left': mBAN_left, 'right': mBAN_right}
    """
    # init dicts
    device_to_subject = {}
    subject_mac_map = {}

    # load the CSV file
    df = pd.read_csv(csv_path, sep=";")
    
    for _, row in df.iterrows():

        # Extract and clean up values
        group = str(row['group']).strip()
        device_num = str(row['device_num']).strip() 
        subject_id = str(row['subject_id']).strip() 
        mBAN_left = str(row['mBAN_left']).strip()
        mBAN_right = str(row['mBAN_right']).strip()

        # Map device_num to subject_id within the group
        device_to_subject[(group, device_num)] = subject_id
        # Map subject_id to MAC addresses within the group
        subject_mac_map[(group, subject_id)] = {'left': mBAN_left, 'right': mBAN_right}

    print("Available device_to_subject keys:", list(device_to_subject.keys()))
    print("Available subject_mac_map keys:", list(subject_mac_map.keys()))
    return device_to_subject, subject_mac_map

# -----------------------------------------------------------------
def discover_dataset_files(device_to_subject: dict, subject_mac_map: dict, base_dir=BASE_DIR):
    """
    For each subactivity of each subject, automatically detects the 2 muscleBAN files,
    using the subject-specific MAC addresses to identify left/right.
    :param device_to_subject: Mapping from (group, device_num) to subject_id.
    :param subject_mac_map: Mapping from (group, subject_id) to left/right MAC addresses.
    :param base_dir: Base directory where the dataset is stored.
    :return: Yields dictionaries with group, subject, day, session, left file, and right file
    """
    all_files_records = []
    # find group folder
    for group in sorted(os.listdir(base_dir)):
        group_path = os.path.join(base_dir, group)
        if not os.path.isdir(group_path):
            print(f"Skipping (not a directory) group: {group_path}")
            continue
        if group.startswith('group'):
            group_num = group[len('group'):]
            print(f"Detected group number: {group_num}")
        else:
            group_num = group
        print(f"Accessing group directory: {group_path}")
        # find sensors folder
        sensors_path = os.path.join(group_path, "sensors")
        if not os.path.isdir(sensors_path):
            print(f"Skipping (no sensors dir) in group: {sensors_path}")
            continue

        print(f"Accessing sensors directory: {sensors_path}")

        # find device folders
        for device_folder in sorted(os.listdir(sensors_path)):
            device_path = os.path.join(sensors_path, device_folder)
            if not os.path.isdir(device_path):
                print(f"Skipping (not a directory) device folder: {device_path}")
                continue
            print(f"Accessing device folder: {device_path}")

            # Extract device_num, e.g., '#001' from 'LibPhys #001'
            m = re.search(r'#\d+', device_folder)
            if not m:
                print(f"Could not extract device_num from folder '{device_folder}'. Skipping.")
                continue
            device_num = m.group(0)

            # Map device_num to subject_id
            key = (str(group_num).strip(), device_num)
            if key not in device_to_subject:
                print(f"Device to subject mapping missing for group '{group_num}', device '{device_num}'. Skipping device.")
                continue
            subject_id = device_to_subject[key]

            # Now use subject_id for the MAC mapping
            mac_key = (str(group_num).strip(), str(subject_id).strip())
            if mac_key not in subject_mac_map:
                print(f"Subject MAC mapping missing for group '{group_num}', subject '{subject_id}'. Skipping subject.")
                continue
            macs = subject_mac_map[mac_key]
            mac_left_norm = macs['left']
            mac_right_norm = macs['right']
            
            # find day folder
            for day in sorted(os.listdir(device_path)):
                day_path = os.path.join(device_path, day)
                if not os.path.isdir(day_path):
                    print(f"Skipping (not a directory) day: {day_path}")
                    continue
                print(f"Accessing day directory: {day_path}")

                # Check session folders
                for session in sorted(os.listdir(day_path)):
                    session_path = os.path.join(day_path, session)
                    if not os.path.isdir(session_path):
                        print(f"Skipping (not a directory) session: {session_path}")
                        continue
                    print(f"Accessing session directory: {session_path}")
                    files = glob.glob(os.path.join(session_path, "*.txt"))
                    mac_to_file = {}

                    # Go through files 
                    for f in files:
                        parts = os.path.basename(f).split("_")
                        if len(parts) < 2: # Not enough parts to extract MAC
                            continue
                        mac = parts[1] 
                        if mac == mac_left_norm:
                            mac_to_file['left'] = f
                        elif mac == mac_right_norm:
                            mac_to_file['right'] = f
                    if 'left' not in mac_to_file:
                        print(f"Left file not found for subject '{subject_id}' in session: {session_path}")
                    if 'right' not in mac_to_file:
                        print(f"Right file not found for subject '{subject_id}' in session: {session_path}")

                    # Append a record if at least one side exists (left or right)
                    rec = {
                        "group": group,
                        "subject": subject_id,
                        "day": day,
                        "session": session,
                        "left": mac_to_file.get("left"),
                        "right": mac_to_file.get("right"),
                    }
                    if rec["left"] or rec["right"]:
                        present_sides = []
                        if rec["left"]: present_sides.append("left")
                        if rec["right"]: present_sides.append("right")
                        print(f"Found {', '.join(present_sides)} file(s) in session: {session_path}")
                        all_files_records.append(rec)

    return all_files_records

        
# -----------------------------------------------------------------
def load_acc_file(path: str, n_bits=N_BITS, g_range=G_RANGE) -> pd.DataFrame:
    """
    Load a muscleBAN accelerometer file, converting ADC values to g.
    :param path: Path to the accelerometer file.
    :param n_bits: Number of bits of the ADC.
    :param g_range: Range of the accelerometer in g.

    :return: DataFrame with columns nSeq, xAcc, yAcc, zAcc in g.
    """
    if os.path.getsize(path) == 0:
        print(f"Warning: {path} is empty. Skipping.")
        return pd.DataFrame(columns=["nSeq"] + ACC_COLUMNS)
    try:
        df = pd.read_csv(path, delimiter="\t", header=None, skiprows=3, usecols=[0,3,4,5])
    except pd.errors.EmptyDataError:
        print(f"Warning: {path} is empty or has no valid data after header. Skipping.")
        return pd.DataFrame(columns=["nSeq"] + ACC_COLUMNS)
    df.columns = ["nSeq"] + ACC_COLUMNS
    for c in ACC_COLUMNS:
        df[c] = adc_to_acceleration(df[c], n_bits, g_range)
    return df


# -----------------------------------------------------------------
def select_random_sample(
    file_records,
    n_files=30,
    min_n_subjects=5,
    min_n_groups=2
):
    """
    Select a representative random sample of file records for quality assessment.

    :param file_records: List of dictionaries, each with keys (group, subject, left, right, ...).
    :param n_files: Number of session records to sample.
    :param min_n_subjects: Minimum number of unique subjects in the sample.
    :param min_n_groups: Minimum number of unique groups in the sample.
    :return: List of sampled file record dicts.
    """
    # 1. Filter out records with missing or empty left/right files
    filtered_records = []
    for rec in file_records:
        left = rec.get("left")
        right = rec.get("right")
        left_ok = left and os.path.exists(left) and os.path.getsize(left) > 0
        right_ok = right and os.path.exists(right) and os.path.getsize(right) > 0
        # Only require at least one side to exist and be non-empty
        if left_ok or right_ok:
            filtered_records.append(rec)
    if len(filtered_records) < n_files:
        print(f"Warning: Only {len(filtered_records)} valid records found, less than requested sample size {n_files}. Returning all valid records.")
        return filtered_records

    # 2. Try sampling until we satisfy subject/group constraints
    for _ in range(1000):  # Up to 1000 attempts
        sample = random.sample(filtered_records, n_files)
        subjects = {rec['subject'] for rec in sample}
        groups = {rec['group'] for rec in sample}
        if len(subjects) >= min_n_subjects and len(groups) >= min_n_groups:
            return sample

    print("Could not sample with desired subject/group diversity. Returning random sample.")
    return random.sample(filtered_records, n_files)