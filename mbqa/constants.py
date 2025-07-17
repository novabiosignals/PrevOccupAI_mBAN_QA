# --- directory ---------------------------------------------------
BASE_DIR = "/Volumes/NO NAME/Backup PrevOccupAI data/jan2023/data"   

# --- subjects info -----------------------------------------------
# Path to the CSV file containing subject MAC addresses and other info.
SUBJECTS_CSV_PATH = "/Users/goncalobarros/Documents/projects/New_MB_QA/Subjects_Info.csv"       

# --- sensor & conversion ----------------------------------------
FS = 1000     # Hz  (real ACC sampling)
G_RANGE = 8  # ±g
N_BITS  = 16

ACC_COLUMNS  = ["xAcc", "yAcc", "zAcc"]
EXPECTED_ACC_RANGE = (-G_RANGE * 9.80665, G_RANGE * 9.80665)  # m/s² for sanity check

# --- quality thresholds -----------------------------------------
SPIKE_THRESHOLD     = 4    # m/s²
MISSING_FRAC_MAX    = 0.01
SPIKE_FRAC_MAX      = 0.01
SATURATION_ATOL     = 1e-6
