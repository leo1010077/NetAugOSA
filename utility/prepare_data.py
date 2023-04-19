import glob
import os
import re
import shutil


def extract_list(data_path, patient_floader, patterns):
    file_list = []
    for patient in patient_floader:
        file_path = data_path + patient
        files = glob.glob(file_path + "/*.wav")
        for file in files:
            for pattern in patterns:
                if re.search(pattern, file):
                    file_list.append([file, patient + file.split('\\')[-1]])
    return file_list


data_path = "../osa_data/"
patient_floader = os.listdir(data_path)

boomsnore_table = ['bs', 'boomsnore', 'BS', 'b']
snore_table = ['s', 'S', 'snore', 'SNORE']
noise_table = ['noise', 'NOISE']
oba_table = ['OBA', 'oba']
hyp_table = ['hyp', 'HYP']

boomsnore_patterns = [re.compile(f'\W{p}') for p in boomsnore_table]
snore_patterns = [re.compile(f'\W{p}') for p in snore_table]
noise_patterns = [re.compile(f'\W{p}') for p in noise_table]
oba_patterns = [re.compile(f'\W{p}') for p in oba_table]
hyp_patterns = [re.compile(f'\W{p}') for p in hyp_table]

boomsnore_lists = extract_list(data_path, patient_floader, boomsnore_patterns)
snore_lists = extract_list(data_path, patient_floader, snore_patterns)
noise_lists = extract_list(data_path, patient_floader, noise_patterns)
oba_lists = extract_list(data_path, patient_floader, oba_patterns)
hyp_lists = extract_list(data_path, patient_floader, hyp_patterns)

destination_path = "../osa_classification1/"

for path in boomsnore_lists:
    shutil.copy(path[0].replace('\\', '/'), destination_path + 'boomsnore/' + path[1])

for path in snore_lists:
    shutil.copy(path[0].replace('\\', '/'), destination_path + 'snore/' + path[1])

for path in noise_lists:
    shutil.copy(path[0].replace('\\', '/'), destination_path + 'noise/' + path[1])

for path in oba_lists:
    shutil.copy(path[0].replace('\\', '/'), destination_path + 'oba/' + path[1])

for path in hyp_lists:
    shutil.copy(path[0].replace('\\', '/'), destination_path + 'hyp/' + path[1])
