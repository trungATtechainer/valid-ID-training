import os
import random
import shutil
from tqdm import tqdm
from multiprocessing import Pool

path = 'data/0/'

all_files = os.listdir(path)

def assign(file):
    if random.random() < 0.1:
        shutil.copyfile(path + file, 'final_data/Valid/0/' + file)
    elif 0.1 < random.random() < 0.15:
        shutil.copyfile(path + file, 'final_data/Test/Valid/0/' + file)
    else:
        shutil.copyfile(path + file, 'final_data/Train/0/' + file)

if __name__ == "__main__":
    pool = Pool(20)
    output = list(tqdm(
        pool.imap(assign, all_files), total=len(all_files), desc="Assigning"))
    pool.terminate()