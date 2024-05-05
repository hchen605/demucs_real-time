import os
import sys
sys.path.append("..")
import wave
import json
import random
import math
import torchaudio
from tqdm import tqdm
from denoiser.dynamic_noiser import apply_echo, apply_speed_and_pitch_change, dynamic_noise_distance_adjustment
from pathlib import Path
import glob
import torchaudio
import torch
from omegaconf import ListConfig

# Function to get the duration of a WAV file in milliseconds
def get_wav_duration_ms(wav_path):
    tnsr, sr = torchaudio.load(wav_path)
    return math.ceil(tnsr.shape[-1] / sr) * 1000

def get_all_wavs_from_folder_path(fp):
    wav_files = []
    #cnt = 0
    # Walk through the directory and its subdirectories
    for root, _, _ in os.walk(fp):
        for file in glob.glob(os.path.join(root, '*.wav')):
            #print(file)
            wav_files.append(file)
    return wav_files

def adjust_audio_length(audio_tensor, target_length):
        """
        Adjusts the length of an audio tensor to a specific target length by clipping or duplicating it.
        If the target length is shorter, it clips a random segment from the audio_tensor.
        Handles both 1D and 2D tensors, assuming 2D tensors are shaped as (channels, samples).

        Parameters:
        - audio_tensor: A PyTorch tensor representing the audio signal, can be 1D or 2D.
        - target_length: The desired length of the audio tensor.

        Returns:
        - A PyTorch tensor representing the adjusted audio signal with the specified target length.
        """
        current_length = audio_tensor.shape[-1]  # Get the current length from the last dimension

        if current_length == target_length:
            # The audio is already the correct length
            return audio_tensor
        elif current_length > target_length:
            # Clip a random segment from the audio to the target length
            max_start_index = current_length - target_length
            start_index = torch.randint(low=0, high=max_start_index + 1, size=(1,)).item()
            if audio_tensor.dim() == 1:
                return audio_tensor[start_index:start_index + target_length]
            else:  # For 2D tensors
                return audio_tensor[:, start_index:start_index + target_length]
        else:
            # Duplicate the audio to reach the target length
            repeat_times = target_length // current_length
            extra_samples = target_length % current_length

            duplicated_audio = audio_tensor.repeat(1, repeat_times) if audio_tensor.dim() == 2 else audio_tensor.repeat(repeat_times)
            if extra_samples > 0:
                if audio_tensor.dim() == 2:
                    duplicated_audio = torch.cat((duplicated_audio, audio_tensor[:, :extra_samples]), dim=1)
                else:
                    duplicated_audio = torch.cat((duplicated_audio, audio_tensor[:extra_samples]))
            
            return duplicated_audio

# Function to generate a list of lists with WAV file paths and their lengths
def generate_wav_info_json(folder_path, noise_path, max_num:int=None):
    wav_info_list = list()
    noise_info_list = list()
    for _, filename in tqdm(enumerate(os.listdir(folder_path)), total=len(os.listdir(folder_path))):
        if len(wav_info_list) == max_num:
            break 

        if filename.endswith(".wav"):
            full_path = os.path.join(folder_path, filename)
            length_ms = get_wav_duration_ms(full_path)
            
            if length_ms < 4000:
                continue

            wav_info_list.append([full_path, length_ms])
            if noise_path:
                noise_info_list.append([os.path.join(noise_path, filename), length_ms])
    return wav_info_list, noise_info_list

""" # Example usage
clean_path = '/mnt/dylan_disk/Vapi_conversations/clipped_user_sequences'
noise_path = "/mnt/dylan_disk/noised_Vapi_clips/noise_clips"
wav_info_list, noise_info_list = generate_wav_info_json(noise_path, clean_path, max_num=10000)

indexes = list(range(len(wav_info_list)))
print('amount: ', len(indexes))
#random.shuffle(indexes)

yt_path = "/mnt/dylan_disk/YT_audio"
yt_files = get_all_wavs_from_folder_path(yt_path)

funcs_w_args = {
            'apply_echo': {
                'delay_ms': [50, 500],
                'decay': [0.05, 0.5],
                'repeats': [0, 3]
            },
            'dynamic_noise_distance_adjustment': {
                'min_distance': 2,
                'max_distance': 6,
                'segments': [1, 5]
            }
}
aug_key_to_func = dict(
            apply_echo=apply_echo, 
            dynamic_noise_distance_adjustment=dynamic_noise_distance_adjustment)
output_dir = '/home/vapi/dataset/dynamic'
results_dir = Path(output_dir)
results_dir.mkdir(parents=True, exist_ok=True)

for file in tqdm(wav_info_list, total=len(wav_info_list)):
    #print(file)
    fname = file[0].split('/')[-1]
    waveform, sample_rate = torchaudio.load(file[0])
    noise_fp = random.choice(yt_files)
    noise_waveform, noise_sample_rate = torchaudio.load(noise_fp)

    # Check if resampling is necessary
    if sample_rate != noise_sample_rate:
        # Create a resample transform
        resample_transform = torchaudio.transforms.Resample(orig_freq=noise_sample_rate, new_freq=sample_rate)
        # Apply the resample transform to the waveform
        noise_waveform = resample_transform(noise_waveform)

    # Adjust noise length to original sequence length
    noise_waveform = adjust_audio_length(audio_tensor=noise_waveform, target_length=waveform.shape[-1])

    # Sample augmentation and sample parameters
    for key, args in funcs_w_args.items():
        sampled_args = dict(sample_rate = sample_rate, noise_audio = noise_waveform)
        for arg_key, val in args.items():
            #print(arg_key, val)
            if isinstance(val, (list,ListConfig)):
                if isinstance(val[0], int) and isinstance(val[1], int):
                    sampled_args[arg_key] = random.randint(*val)
                else:
                    sampled_args[arg_key] = random.uniform(*val)
            else:
                sampled_args[arg_key] = val
            #print(sampled_args[arg_key])
        #print(sampled_args)
        noise_waveform = aug_key_to_func[key](**sampled_args)

    if noise_waveform.shape[0] > 1:
        noise_waveform = noise_waveform.mean(dim=0)

    merged_waveform = torch.concat([waveform, 0.5*noise_waveform.unsqueeze(0)], dim=0)
    torchaudio.save(output_dir+'/'+fname, merged_waveform.mean(dim=0).unsqueeze(0), sample_rate)

 """
output_dir = '/home/vapi/dataset/dynamic'
clean_path = '/mnt/dylan_disk/Vapi_conversations/clipped_user_sequences'
wav_info_list, noise_info_list = generate_wav_info_json(output_dir, clean_path)

indexes = list(range(len(wav_info_list)))
split = int(len(indexes)*0.8)
train_indexes = indexes[:split]
test_indexes = indexes[split:]

for name, idx_list in (("tr", train_indexes), ("ts", test_indexes)):
    clean = [wav_info_list[i] for i in idx_list]
    noise = [noise_info_list[i] for i in idx_list]

    # Define the folder path
    folder_path = f"/home/vapi/ML-denoiser/egs/large_dyn/{name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    clean_file_path = os.path.join(folder_path, "clean.json")
    with open(clean_file_path, "w") as file:
        json.dump(noise, file, indent=4)

    noisy_file_path = os.path.join(folder_path, "noisy.json")
    with open(noisy_file_path, "w") as file:
        json.dump(clean, file, indent=4)