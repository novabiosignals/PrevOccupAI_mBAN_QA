# main libraries
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


# internal libraries
from .constants import (ACC_COLUMNS, EXPECTED_ACC_RANGE, SPIKE_THRESHOLD, SATURATION_ATOL, N_BITS, G_RANGE, FS)
from .loader import load_acc_file
from .adc_a import adc_to_acceleration
from .viz import (plot_missing_data, plot_out_of_range, plot_flatlines,
                  plot_spikes, plot_variance)
from .mBANs_noise import min_var_limit

# --- QUALITY ASSESSMENT ---
def assess_acc_quality(acc_df: pd.DataFrame, generate_reports: bool = True) -> dict:
    """
    Assess quality metrics for each accelerometer (ACC) axis in a DataFrame.

    This function performs a robust quality assessment for each ACC axis,
    including checks for missing data, out-of-range values, flatlines,
    spikes, variance, and analog saturation. All thresholds and parameters
    should be defined in the constants module or passed through the function signature.

    :param acc_df: DataFrame containing columns 'nSeq' and all ACC axes (e.g., 'xAcc', 'yAcc', 'zAcc').

    :returns: 
        Dictionary mapping each ACC axis to a nested dictionary of quality metrics:
            - frac_missing (float): Fraction of missing samples.
            - out_of_range_frac (float): Fraction of samples outside expected range.
            - flatline_samples_frac (float): Fraction of samples in flatline segments.
            - spikes_frac (float): Fraction of detected spikes.
            - variance (float): Variance of the signal.
            - saturation_frac (float): Fraction of saturated values.
            - status (str): 'GOOD' or 'BAD' according to QA thresholds.
    """

    quality_report = {}
    nseq = acc_df['nSeq']
    nseq = nseq.to_numpy() if isinstance(nseq, pd.Series) else np.array(nseq)

    for col in ACC_COLUMNS:
        report = {}
        data = acc_df[col]
        data = data.to_numpy() if isinstance(data, pd.Series) else np.array(data)

        # 1. Missingness
        frac_missing, n_missing, missing_indices, miss_list = missing_data(nseq)
        report['frac_missing'] = frac_missing
        
        # 2. Out-of-range
        out_of_range_frac, n_out_of_range, out_of_range_mask = detect_out_of_range(
            data, EXPECTED_ACC_RANGE
        )
        report['out_of_range_frac'] = out_of_range_frac
        
        # 3. Flatlines (all durations in seconds for consistency)
        flatline_frac, n_flatline, flatline_mask = detect_flatlines(
            data, 
            fs=100, 
            flatline_duration=10,  # seconds; set in constants for clarity
            atol= SATURATION_ATOL  # absolute tolerance for flatline detection
        )
        report['flatline_samples_frac'] = flatline_frac
        
        # 4. Spikes
        spikes_dict, spike_properties, spikes_frac = detect_spikes_with_peaks(
            data,
            fs=FS,
            min_duration=0.03,  # seconds, physiological minimum
            height=SPIKE_THRESHOLD,
            prominence=None,
            distance=None,
            startup_n=30,
            startup_thresh=10.0,
            max_width_samples=3,
            local_std_thresh=2.0,
            local_std_window_ms=100
        )
        report['spikes_frac'] = spikes_frac
        spike_indices = spikes_dict['peaks']
        
        # 5. Saturation only
        saturation_frac = calc_saturation(data, n_bits=N_BITS, g_range=G_RANGE)
        report['saturation_frac'] = saturation_frac

        # 6. Variance
        var, is_below_threshold = calc_variance(data, var_limit_axis=min_var_limit)
        report['variance'] = var
        report['variance_below_threshold'] = is_below_threshold

    

        if generate_reports:
            # Plot out-of-range values
            plot_out_of_range(data, 
                out_of_range_mask, 
                expected_range=EXPECTED_ACC_RANGE, 
                group=acc_df['group'].iloc[0], 
                subject=acc_df['subject'].iloc[0], 
                session=acc_df['session'].iloc[0], 
                side=acc_df['side'].iloc[0], 
                axis=col, 
                outdir='reports'
            )
            # Plot missing data
            plot_missing_data(
                nseq,
                missing_indices,
                miss_list,
                frac_missing,
                n_missing,
                group=acc_df['group'].iloc[0],
                subject=acc_df['subject'].iloc[0],
                session=acc_df['session'].iloc[0],
                side=acc_df['side'].iloc[0],
                axis=col,
                outdir='reports'
            )
        
            
            # Plot flatlines
            plot_flatlines(data, 
                problem_flatline_mask=flatline_mask, 
                atol= SATURATION_ATOL, 
                group=acc_df['group'].iloc[0], 
                subject=acc_df['subject'].iloc[0], 
                session=acc_df['session'].iloc[0], 
                side=acc_df['side'].iloc[0], 
                axis=col, 
                outdir='reports'
            )
            
            # Plot spikes
            plot_spikes(data, 
                peaks_idx=spike_indices, 
                height=SPIKE_THRESHOLD, 
                group=acc_df['group'].iloc[0], 
                subject=acc_df['subject'].iloc[0], 
                session=acc_df['session'].iloc[0], 
                side=acc_df['side'].iloc[0], 
                axis=col, 
                outdir='reports'
            )
            # Plot variance
            plot_variance(data, 
                group=acc_df['group'].iloc[0], 
                subject=acc_df['subject'].iloc[0], 
                session=acc_df['session'].iloc[0], 
                side=acc_df['side'].iloc[0], 
                axis=col, 
                outdir='reports'
            )
        

        # 6. Overall status (set thresholds according to scientific and empirical criteria)
        # Note: spikes_frac is a fraction, not the spike indices array
        if (
            report['flatline_samples_frac'] > 0.01 or
            report['out_of_range_frac'] > 0 or
            report['spikes_frac'] > 0.01 or
            report['frac_missing'] > 0.01 or
            report['variance_bellow_threshold'] == True or
            report['saturation_frac'] > 0.01
        ):
            report['status'] = 'BAD'
            # Identify which threshold(s) caused the BAD status
            failed_criteria = []
            if report['flatline_samples_frac'] > 0.01:
                failed_criteria.append(f"flatline_samples_frac ({report['flatline_samples_frac']:.4f} > 0.01)")
            if report['out_of_range_frac'] > 0:
                failed_criteria.append(f"out_of_range_frac ({report['out_of_range_frac']:.4f} > 0)")
            if report['spikes_frac'] > 0.01:
                failed_criteria.append(f"spikes_frac ({report['spikes_frac']:.4f} > 0.01)")
            if report['frac_missing'] > 0.01:
                failed_criteria.append(f"frac_missing ({report['frac_missing']:.4f} > 0.01)")
            if report['variance_below_threshold']:
                failed_criteria.append(
                    f"variance ({report['variance']:.4e} < {min_var_limit:.4e})"
                )
            if report['saturation_frac'] > 0.01:
                failed_criteria.append(f"saturation_frac ({report['saturation_frac']:.4f} > 0.01)")
            report['extra_info'] = "; ".join(failed_criteria)
        else:
            report['status'] = 'GOOD'

        quality_report[col] = report

    return quality_report

# --- COMPARE TWO FILES (LEFT/RIGHT) ---
def compare_two_acc_files(file_left, file_right) -> list[dict]:
    """
    Compare the quality assessment metrics of two MuscleBAN accelerometer files (left and right).

    This function loads two accelerometer data files, performs signal quality assessment
    for each using ``assess_acc_quality``, and returns a list of results suitable for
    plotting or reporting. Each result contains the side ("left" or "right"),
    the session identifier, the file path, and a dictionary of quality metrics for each axis.

    :param file_left: Path to the MuscleBAN file for the left side.
    :param file_right: Path to the MuscleBAN file for the right side.

    :returns:
        results (list[dict]): A list with two dictionaries:
            - Each dictionary contains:
                - 'side' (str): 'left' or 'right'
                - 'session' (str): Session identifier derived from the filename
                - 'file' (str): Original file path
                - 'quality' (dict): Nested quality metrics for each accelerometer axis
    """
    acc_left = load_acc_file(file_left)
    quality_left = assess_acc_quality(acc_left)

    acc_right = load_acc_file(file_right)
    quality_right = assess_acc_quality(acc_right)

    print(f"Checked LEFT:  {[quality_left[axis]['status'] for axis in ACC_COLUMNS]}")
    print(f"Checked RIGHT: {[quality_right[axis]['status'] for axis in ACC_COLUMNS]}")

    results = [
        {'side': 'left', 'session': os.path.basename(file_left), 'file': file_left, 'quality': quality_left},
        {'side': 'right', 'session': os.path.basename(file_right), 'file': file_right, 'quality': quality_right}
    ]
    return results

# ---------------------------
# Quality Assessment Functions
# ---------------------------

# --- MISSING DATA ---
def missing_data(nseq: np.ndarray) -> Tuple[float, int, np.ndarray, list[int]]:
    """
    Calculate the fraction and number of missing samples in a sequence of sample numbers.

    :param nseq: Sequence of sample numbers (monotonically increasing).

    :returns:
        frac_missing (float): Fraction of missing samples relative to observed sequence length.
        n_missing (int): Total number of missing samples.
        missing_indices (np.ndarray): Indices where missing samples occur.
        miss_list (list[int]): List of counts of missing samples at each missing index.
    """
    if nseq is None or len(nseq) == 0:
        print("Warning: Empty nseq in missing_data")
        return float('nan'), 0, np.array([], dtype=int), []
    diff_array = np.diff(nseq)
    print(len(diff_array) == len(nseq) - 1)  # Ensure diff_array has the correct length

    missing_indices = np.where(diff_array > 1)[0]
    print(f"Missing indices found: {missing_indices}")
    miss_list = []
    num_occorr = 0
    for el in diff_array:
        if el > 1:
            miss_list.append(el - 1)
            num_occorr += 1
    n_missing = np.sum(miss_list)
    frac_missing = n_missing / len(nseq) if len(nseq) > 0 else float('nan')
    
    print(f"Fraction of missing samples: {frac_missing:.4f}, Total missing samples: {n_missing}")

    for idx, gap in zip(missing_indices, miss_list):
        print(f"Missing at index {idx} with gap of {gap} samples")


    return frac_missing, n_missing, missing_indices, miss_list

# --- SATURATION ONLY ---
def calc_saturation(
    data: np.ndarray, 
    n_bits: int = 16, 
    g_range: int = 8
) -> float:
    """
    Calculate the analog saturation fraction for an accelerometer axis.

    :param data: The data vector for a single accelerometer axis (e.g., xAcc).
    :param n_bits: Number of bits for the sensor ADC (default: 16).
    :param g_range: Accelerometer range in g (default: 8 for ±8g).

    :returns: 
        - saturation_frac (float): Fraction of samples that are at the ADC saturation limit (max or min value).
    """
    if data is None or len(data) == 0:  # Check for empty data
        print("Warning: Empty data provided for saturation calculation.")
        return float('nan')

    # Analog saturation values (ADC limits for given bit depth and range)
    saturation_values = [
        adc_to_acceleration(2**(n_bits-1)-1, n_bits=n_bits, g_range=g_range),   # max positive value
        adc_to_acceleration(-2**(n_bits-1), n_bits=n_bits, g_range=g_range)     # max negative value
    ]
    # Count saturations
    saturation = (
        np.isclose(data, saturation_values[0], atol=1e-4) |
        np.isclose(data, saturation_values[1], atol=1e-4)
    ).sum()
    saturation_frac = saturation / len(data) if len(data) > 0 else float('nan')

    return saturation_frac

# --- SPIKE DETECTION ---
def detect_spikes_with_peaks(
    data: np.ndarray,
    fs: int = 1000,                    # Sampling frequency (Hz)
    min_duration: float = 0.03,        # Minimum plausible physiological spike duration (s)
    height: float = 4.0,               # m/s²; threshold
    prominence: 'Optional[float]' = None,
    distance: 'Optional[int]' = None,
    startup_n: int = 30,               # Number of first samples to check for artifacts
    startup_thresh: float = 10.0,       # Threshold for startup artifacts
    max_width_samples: int = 3,         # Maximum width of peak in samples to consider as spike artifact
    local_std_thresh: float = 2.0,       # Threshold for local std deviation to exclude peaks in high activity regions
    local_std_window_ms: int = 100       # Window size (in milliseconds) for local standard deviation calculation
) -> tuple[dict, dict, float]:
    """
    Detect spike artifacts using :func:`scipy.signal.find_peaks`,  
    with width based on the sensor's sampling frequency. Also checks the first N samples for high-amplitude
    start-up artifacts (which might not be detected by peak finding).

    After initial peak detection, filters peaks by their widths (using :func:`scipy.signal.peak_widths`) to keep only narrow peaks,
    and excludes peaks occurring within periods of high local activity defined by a local standard deviation threshold.

    Parameters
    ----------
    data : np.ndarray
        The 1D time series (e.g., accelerometer axis).
    fs : int
        Sampling frequency in Hz.
    min_duration : float
        Minimum duration of a physiological event related to human activity in seconds.
    height : float
        Minimum height to be considered a spike (absolute, in ACC units).
    prominence : float or None
        Minimum prominence of a spike (see scipy docs).
    distance : int or None
        Minimum horizontal distance (in samples) between neighboring peaks.
    startup_n : int
        Number of first samples to check for start-up artifacts (default: 30).
    startup_thresh : float
        Amplitude threshold for start-up artifact detection (default: 20.0 m/s²).
    max_width_samples : int
        Maximum peak width in samples to consider a peak as a spike artifact (default: 3).
    local_std_thresh : float
        Threshold for local standard deviation to exclude peaks occurring in high activity regions (default: 2.0).
    local_std_window_ms : int
        Window size (in milliseconds) for local standard deviation calculation (default: 100 ms).

    Returns
    -------
    spikes_dict : dict
        Dictionary with keys:
        - 'peaks': np.ndarray of indices of detected spikes (including start-up artifacts).
        - 'startup': np.ndarray of indices of detected start-up artifacts.
    properties : dict
        Properties returned by :func:`scipy.signal.find_peaks` (for analysis).
    spikes_fraction : float
        Fraction of detected spikes relative to total samples.
    """

    width_samples = int(np.ceil(min_duration * fs))
    if width_samples < 1:
        raise ValueError(
            f"Peak width is less than 1 sample: computed width_samples={width_samples} "
            f"from min_duration={min_duration} and fs={fs}. "
            "Increase min_duration or fs."
        )
    # Find peaks using scipy's find_peaks
    peaks_idx, properties = find_peaks(
        np.abs(data),
        height=height,
        prominence=prominence,
        width=width_samples,
        distance=distance,
    )
    # Calculate widths of detected peaks
    results_half = peak_widths(np.abs(data), peaks_idx, rel_height=0.5)
    peak_widths_samples = results_half[0].astype(int)  # Widths in samples

    # Filter peaks by max_width_samples
    narrow_peaks_mask = peak_widths_samples <= max_width_samples
    filtered_peaks_idx = peaks_idx[narrow_peaks_mask]

    # Compute local std deviation around each peak (±local_std_window_ms samples)
    half_window = int((local_std_window_ms / 1000.0) * fs)
    local_stds = []
    for peak in filtered_peaks_idx:
        start = max(0, peak - half_window)
        end = min(len(data), peak + half_window + 1)
        local_std = np.std(data[start:end])
        local_stds.append(local_std)
    local_stds = np.array(local_stds)

    # Filter peaks by local std deviation threshold
    local_std_mask = local_stds <= local_std_thresh
    final_peaks_idx = filtered_peaks_idx[local_std_mask]

    # Manually check first N samples for startup artifacts
    startup_artifacts = np.where(np.abs(data[:startup_n]) > startup_thresh)[0]
    print(f"Detected {len(startup_artifacts)} startup artifact(s) in first {startup_n} samples.")

    # Combine indices and ensure uniqueness
    all_spikes = np.unique(np.concatenate([final_peaks_idx, startup_artifacts]))
    spikes_fraction = len(all_spikes) / len(data) if len(data) > 0 else float('nan')
    print(f"Detected {len(all_spikes)} spikes ({spikes_fraction:.4f} of total samples)")
    return {'peaks': all_spikes, 'startup': startup_artifacts}, properties, spikes_fraction

# --- FLATLINE DETECTION ---
def detect_flatlines(
    data: np.ndarray,
    fs: int = FS,  
    flatline_duration: float = 10,  # Minimum flatline duration in seconds
    atol: float = 1e-6
) -> tuple[float, int, np.ndarray]:
    """
    Detect flatline segments in a 1D time series, with a minimum flatline duration defined in milliseconds.

    :param data: The 1D signal (e.g., ACC axis).
    :param fs: Sampling frequency in Hz.
    :param flatline_duration_ms: Minimum flatline segment duration (in milliseconds) to be flagged. Deault is 10 seconds (why and paper).
    :param atol: Absolute tolerance for "no change" (flatness). Default is 1e-6.

    :returns: Tuple containing:
        - flatline_fraction (float): Fraction of samples involved in problematic flatlines.
        - n_flatline_samples (int): Total number of flatline samples above threshold.
        - flatline_mask (np.ndarray): Boolean array marking flatline samples above threshold.
    """
    if data is None or len(data) == 0:
        print("Warning: Empty data in detect_flatlines")
        return float('nan'), 0, np.array([], dtype=bool)
    # Compute the minimum number of samples corresponding to the desired duration
    min_flatline_samples = int(np.ceil(flatline_duration * fs))
    if min_flatline_samples < 1:
        raise ValueError(
            f"Flatline threshold is less than 1 sample: computed min_flatline_samples={min_flatline_samples} "
            f"from flatline_duration_ms={flatline_duration} and fs={fs}. "
            "Increase flatline_duration_ms or fs."
        )
    
    # Find flat segments (difference between consecutive points is less than atol)
    flat_mask = np.abs(np.diff(data)) < atol
    # Pad to align with original array length
    flat_mask = np.insert(flat_mask, 0, False)

    # Convert the boolean flat_mask to 0/1 integer Series for easier group operations
    flatline_locs = pd.Series(flat_mask).astype(int)

    # Find where the state (flat/not flat) changes in the sequence
    state_change = flatline_locs != flatline_locs.shift()

    # Create unique group labels for each contiguous run (e.g., all flat, all not flat, etc.)
    group_labels = state_change.cumsum()

    # For each sample, compute the length of its contiguous run
    # (Each sample in a flat run will get the run length; samples in not-flat runs get zero)
    run_lengths = flatline_locs.groupby(group_labels).transform('sum')

    # Identify samples that are in a flat run AND whose run is at least the minimum required length
    problem_flatline_mask = (run_lengths >= min_flatline_samples) & (flatline_locs == 1)

    # Count the total number of problematic flatline samples
    n_flatline_samples = problem_flatline_mask.sum()
    print(f"Detected {n_flatline_samples} problematic flatline samples out of {len(data)} total samples.")

    # Calculate the fraction of samples that are part of problematic flatlines
    flatline_fraction = n_flatline_samples / len(data) if len(data) > 0 else float('nan')

    return flatline_fraction, n_flatline_samples, np.asarray(problem_flatline_mask.values)

# --- OUT OF RANGE DETECTION ---
def detect_out_of_range(
    data: np.ndarray,
    expected_range: tuple[float, float]
) -> tuple[float, int, np.ndarray]:
    """
    Detect samples in a signal that fall outside an expected physical range.

    :param data: The input signal (e.g., accelerometer axis).
    :param expected_range: Tuple specifying the expected (min, max) value range.

    :returns:
        - out_of_range_fraction (float): Fraction of samples outside the expected range.
        - n_out_of_range (int): Number of out-of-range samples.
        - out_of_range_mask (np.ndarray): Boolean mask marking out-of-range samples.
    """
    if data is None or len(data) == 0:
        print("Warning: Empty data in detect_out_of_range")
        return float('nan'), 0, np.array([], dtype=bool)
    out_of_range_mask = (data < expected_range[0]) | (data > expected_range[1])
    n_out_of_range = out_of_range_mask.sum()
    out_of_range_fraction = n_out_of_range / len(data) if len(data) > 0 else float('nan')
    return out_of_range_fraction, n_out_of_range, out_of_range_mask

# Function to calculate variance and identify low variance axes
def calc_variance(data, var_limit_axis=min_var_limit):
    """
    Calculate variance for a single axis and check if it is below the threshold.

    :param data: 1D input array for a single axis.
    :param var_limit_axis: Variance threshold to determine low-variance.
    :return: Tuple containing:
        - var: Variance (float).
        - is_below_threshold: Boolean indicating if variance is below the threshold.
    """
    var = np.var(data, ddof=0)
    is_below_threshold = var < var_limit_axis # output is True if variance is below the threshold
    print("Variance:", var)
    print('Is it below threshold?', is_below_threshold)
    return var, is_below_threshold
