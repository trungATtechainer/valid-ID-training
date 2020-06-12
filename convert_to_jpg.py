import cv2
import os
from multiprocessing import Pool
from tqdm import tqdm

all_files = os.listdir('data/0')

def convert(file):
    im = cv2.imread('data/0/' + file)
    name = file.split('.')[0]
    cv2.imwrite('data/0_final/' + name + '.jpg', im)

pool = Pool(8)
output = list(tqdm(
    pool.imap(convert, all_files), total=len(all_files), desc="converting"))
pool.terminate()