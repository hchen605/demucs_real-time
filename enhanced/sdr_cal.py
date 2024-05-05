import os
import numpy as np
import soundfile as sf

def calculate_sdr(clean_signal, estimated_signal):
    """
    Calculate the Signal-to-Distortion Ratio (SDR).

    Parameters:
    - clean_signal: numpy array, the original clean audio signal.
    - estimated_signal: numpy array, the estimated audio signal.

    Returns:
    - SDR: float, the calculated SDR value in dB.
    """
    # Ensure the signals are aligned in length
    min_len = min(len(clean_signal), len(estimated_signal))
    clean_signal = clean_signal[:min_len]
    estimated_signal = estimated_signal[:min_len]

    # Calculate signal power and error power
    signal_power = np.sum(clean_signal ** 2)
    error_power = np.sum((clean_signal - estimated_signal) ** 2)

    # Calculate SDR
    sdr = 10 * np.log10(signal_power / error_power)
    return sdr

def calculate_sdr_for_folder(clean_folder, estimated_folder):
    """
    Calculate the SDR for all pairs of clean and estimated signals in the given folders.

    Parameters:
    - clean_folder: str, path to the folder containing clean signals.
    - estimated_folder: str, path to the folder containing estimated signals.
    """
    # Ensure both folders exist
    if not os.path.exists(clean_folder) or not os.path.exists(estimated_folder):
        print("One of the folders does not exist.")
        return
    
    sdr_values = []

    # Iterate over clean files
    for file in os.listdir(clean_folder):
        clean_file_path = os.path.join(clean_folder, file)
        fname = file[:-4]
        fname = fname + '_enhanced.wav'
        estimated_file_path = os.path.join(estimated_folder, fname)

        # Ensure the estimated file exists
        if not os.path.exists(estimated_file_path):
            print(f"No estimated file matches {file}")
            continue

        # Read the clean and estimated signals
        clean_signal, _ = sf.read(clean_file_path)
        estimated_signal, _ = sf.read(estimated_file_path)
        if clean_signal.shape[1] == 2:
            clean_signal = (clean_signal[:,0] + clean_signal[:,1])/2
            
        # Calculate SDR
        sdr = calculate_sdr(clean_signal, estimated_signal)
        sdr_values.append(sdr)
        print(f"SDR for {file}: {sdr} dB")

    # Optionally, compute average SDR
    if sdr_values:
        average_sdr = sum(sdr_values) / len(sdr_values)
        print(f"Average SDR: {average_sdr} dB")

# Example usage
clean_folder = "../dataset/testset_20/clean"
estimated_folder = "testset_enhanced_remix_l2_dry_0p2"
calculate_sdr_for_folder(clean_folder, estimated_folder)
