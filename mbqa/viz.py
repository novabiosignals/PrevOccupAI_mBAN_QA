# main libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from typing import Optional
from glob import glob
import re

# Auxiliar libraries
from .constants import ACC_COLUMNS, EXPECTED_ACC_RANGE
from .loader import load_acc_file


# ---------------------------
# Plotting Functions
# ---------------------------

def plot_missing_data(nseq, missing_indices, miss_list, frac_missing, n_missing, group, subject, session, side, axis, outdir):
    """
    Plot missing data points in a sample sequence and save the figure as an image.

    Parameters
    ----------
    nseq : np.ndarray or list
        Sequence numbers of the samples.
    missing_indices : list or np.ndarray
        Indices where samples are missing.
    group : str or int
        Group identifier for the session.
    subject : str or int
        Subject identifier.
    session : str
        Session identifier (e.g., date or timestamp).
    side : str
        Sensor side ('left' or 'right').
    axis : str
        Axis name (e.g., 'xAcc').
    outdir : str
        Directory where the output image will be saved.

    Returns
    -------
    None
        The function saves the generated plot to disk and does not return anything.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(len(nseq)), nseq, label='nSeq', color='blue')
    # Only add the label once for the legend
    label_added = False
    for idx, gap in zip(missing_indices, miss_list):
        label = 'Missing region' if not label_added else None
        ax.axvspan(idx, idx + gap, color='red', alpha=0.2, label=label)
        if not label_added:
            label_added = True
    ax.set_title(
        f"nSeq with Missing Samples Highlighted\n"
        f"Fraction Missing: {frac_missing:.4f}, Total Missing: {n_missing}"
    )
    ax.set_xlabel("Index")
    ax.set_ylabel("nSeq Value")
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(outdir, f"missing_data_group{group}_subject{subject}_session{session}_{side}_{axis}.png")
    fig.savefig(fname)
    plt.close(fig)

def plot_variance(
    data, group, subject, session, side, axis, outdir, 
    min_var_limit: Optional[float] = None
):
    """
    Plot accelerometer data and optionally a horizontal line for the minimum variance threshold.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Accelerometer signal data for one axis.
    group : str or int
        Group identifier for the session.
    subject : str or int
        Subject identifier.
    session : str
        Session identifier (e.g., date or timestamp).
    side : str
        Sensor side ('left' or 'right').
    axis : str
        Axis name (e.g., 'xAcc').
    outdir : str
        Directory where the output image will be saved.
    min_var_limit : float, optional
        Minimum variance threshold to plot (from mBANs_noise).

    Returns
    -------
    None
        The function saves the generated plot to disk and does not return anything.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Accelerometer Data', marker='o', markersize=2, alpha=0.7)
    if min_var_limit is not None:
        plt.axhline(y=min_var_limit, color='red', linestyle='--', label=f'Min Variance Threshold ({min_var_limit:.2e})')
    plt.title('Accelerometer Data with Variance Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Accelerometer Value (acceleration in m/s²)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = f"{outdir}/variance_group{group}_subject{subject}_session{session}_{side}_{axis}.png"
    plt.savefig(fname)
    plt.close()

def plot_spikes(data, peaks_idx, height, group, subject, session, side, axis, outdir):
    """
    Plot detected spike artifacts in accelerometer data and save the figure as an image.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Accelerometer signal data for one axis.
    peaks_idx : np.ndarray or list
        Indices of detected spikes in the data.
    height : float
        Minimum height threshold for spikes.
    group : str or int
        Group identifier for the session.
    subject : str or int
        Subject identifier.
    session : str
        Session identifier (e.g., date or timestamp).
    side : str
        Sensor side ('left' or 'right').
    axis : str
        Axis name (e.g., 'xAcc').
    outdir : str
        Directory where the output image will be saved.

    Returns
    -------
    None
        The function saves the generated plot to disk and does not return anything.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Accelerometer Data', marker='o', markersize=2, color='tab:blue', alpha=0.7)
    if len(peaks_idx) > 0:
        plt.scatter(peaks_idx, np.array(data)[peaks_idx], color='red', label='Detected Spikes', s=60, zorder=3, marker='x')
        for idx in peaks_idx:
            plt.annotate(f"{idx}", (idx, data[idx]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='red')
    plt.axhline(y=height, color='red', linestyle='--', label='Spike Threshold')
    plt.title(f"Detected Spikes in Accelerometer Data\nGroup: {group}, Subject: {subject}, Session: {session}, {side.capitalize()} {axis}")
    plt.xlabel('Sample Index')
    plt.ylabel('Accelerometer Value (acceleration in m/s²)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = f"{outdir}/spikes_group{group}_subject{subject}_session{session}_{side}_{axis}.png"
    plt.savefig(fname)
    plt.close()

def plot_flatlines(data, problem_flatline_mask, atol, group, subject, session, side, axis, outdir):
    """
    Plot detected flatline regions in accelerometer data and save the figure as an image.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Accelerometer signal data for one axis.
    problem_flatline_mask : np.ndarray or pd.Series of bool
        Boolean mask indicating samples identified as flatline.
    atol : float
        Absolute tolerance threshold used for flatline detection.
    group : str or int
        Group identifier for the session.
    subject : str or int
        Subject identifier.
    session : str
        Session identifier (e.g., date or timestamp).
    side : str
        Sensor side ('left' or 'right').
    axis : str
        Axis name (e.g., 'xAcc').
    outdir : str
        Directory where the output image will be saved.
    Returns
    -------
    None
        The function saves the generated plot to disk and does not return anything.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Accelerometer Data', marker='o', markersize=2)
    flat_indices = np.where(problem_flatline_mask)[0]
    if len(flat_indices) > 0:
        plt.plot(flat_indices, data[problem_flatline_mask], 'rx', label='Detected Flatlines', markersize=8)
    plt.title('Detected Flatlines in Accelerometer Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Accelerometer Value (acceleration in m/s²)')
    plt.axhline(y=atol, color='red', linestyle='--', label='Tolerance Threshold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"{outdir}/flatlines_group{group}_subject{subject}_session{session}_{side}_{axis}.png"
    plt.savefig(fname)
    plt.close()

def plot_out_of_range(data, out_of_range_mask, expected_range, group, subject, session, side, axis, outdir):
    """
    Plot out-of-range samples in accelerometer data and save the figure as an image.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Accelerometer signal data for one axis.
    out_of_range_mask : np.ndarray or pd.Series of bool
        Boolean mask indicating samples that are out of the expected physical range.
    expected_range : tuple of float
        Minimum and maximum expected physical values (e.g., (min_g, max_g)).
    group : str or int
        Group identifier for the session.
    subject : str or int
        Subject identifier.
    session : str
        Session identifier (e.g., date or timestamp).
    side : str
        Sensor side ('left' or 'right').
    axis : str
        Axis name (e.g., 'xAcc').
    outdir : str
        Directory where the output image will be saved.

    Returns
    -------
    None
        The function saves the generated plot to disk and does not return anything.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Accelerometer Data', marker='o', markersize=2)
    out_indices = np.where(out_of_range_mask)[0]
    if len(out_indices) > 0:
        plt.plot(out_indices, data[out_of_range_mask], 'rx', label='Out of Range Samples', markersize=8)
    plt.title('Out of Range Detection in Accelerometer Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Accelerometer Value (acceleration in m/s²)')
    plt.axhline(y=expected_range[0], color='red', linestyle='--', label='Expected Min Range')
    plt.axhline(y=expected_range[1], color='blue', linestyle='--', label='Expected Max Range')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"{outdir}/out_of_range_group{group}_subject{subject}_session{session}_{side}_{axis}.png"
    plt.savefig(fname)
    plt.close()

# ---------------------------
# Exporting Functions
# ---------------------------

def export_session_report_to_pdf(outdir: str, session_prefix: str, pdf_name: Optional[str] = None) -> None:
    """
    Exports all PNG plots related to a specific session into a single PDF file.

    Parameters
    ----------
    outdir : str
        Directory containing the PNG plot files for the session.
    session_prefix : str
        Unique prefix to match plot files for this session (e.g., 'group1_subject26_session2022-05-02_10-00-01').
    pdf_name : str or None, optional
        Name of the output PDF file. If None, defaults to '{session_prefix}_report.pdf' in outdir.

    Returns
    -------
    None
        The function saves a PDF report to disk and does not return anything.
    """
    pattern = f"{outdir}/*{session_prefix}*.png"
    png_files = sorted(glob(pattern))
    if not png_files:
        print(f"[WARN] No plot PNGs found for session '{session_prefix}' in {outdir}")
        return
    if pdf_name is None:
        pdf_name = f"{outdir}/{session_prefix}_report.pdf"
    with PdfPages(pdf_name) as pdf:
        for png in png_files:
            fig = plt.figure()
            img = plt.imread(png)
            plt.imshow(img)
            plt.axis('off')
            # Optional: Add a page title from the filename
            base = os.path.basename(png)
            title = re.sub(r"(_group.*_|\.png)", "", base)
            plt.title(title)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"✓ PDF report saved: {pdf_name}")