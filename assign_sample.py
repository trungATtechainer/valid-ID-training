import os
from shutil import copyfile, move
import random
from tqdm import tqdm

for dataset in ['Train/', 'Test/Valid/', 'Valid/']:
    for classes in ['0/', 'front/', 'back/']:
        path = 'data/' + dataset + classes
        output_path = 'sample/' + dataset + classes
        all_files = os.listdir(path)
        for file in tqdm(all_files):
            if random.random() <  0.05:
                copyfile(path + file, output_path  + file)