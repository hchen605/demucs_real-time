import os
import wave
import json
import random
import math
import torchaudio
from tqdm import tqdm

# Function to get the duration of a WAV file in milliseconds
def get_wav_duration_ms(wav_path):
    tnsr, sr = torchaudio.load(wav_path)
    return math.ceil(tnsr.shape[-1] / sr) * 1000

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

# Example usage
#folder_path = '/mnt/dylan_disk/Vapi_conversations/clipped_user_sequences'
#noise_path = "/mnt/dylan_disk/noised_Vapi_clips/noise_clips"
folder_path = '/home/vapi/LibriMix/librispeech/Libri2Mix/wav16k/min/train-100/s1'
noise_path = '/home/vapi/LibriMix/librispeech/Libri2Mix/wav16k/min/train-100/mix_clean'
wav_info_list, noise_info_list = generate_wav_info_json(noise_path, folder_path, max_num=None)

indexes = list(range(len(wav_info_list)))
#print('amount: ', len(indexes))
random.shuffle(indexes)

#mid_index = len(indexes) // 2
#first_half_indexes = indexes[:mid_index]
#second_half_indexes = indexes[mid_index:]

split = int(len(indexes)*0.8)
train_indexes = indexes[:split]
test_indexes = indexes[split:]

for name, idx_list in (("tr", train_indexes), ("ts", test_indexes)):
    clean = [wav_info_list[i] for i in idx_list]
    noise = [noise_info_list[i] for i in idx_list]

    # Define the folder path
    folder_path = f"/home/vapi/ML-denoiser/egs/librimix2/{name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    clean_file_path = os.path.join(folder_path, "clean.json")
    with open(clean_file_path, "w") as file:
        json.dump(noise, file, indent=4)

    noisy_file_path = os.path.join(folder_path, "noisy.json")
    with open(noisy_file_path, "w") as file:
        json.dump(clean, file, indent=4)