import os
import cv2
import json
import random
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import glob
import sys

all_files = glob.glob('data/1/*')

def crop(file):
    image = cv2.imread(file)
    name = os.path.basename(file).split('.')[0]

    with open("data/json/{}.json".format(name)) as json_file:
        data = json.load(json_file)

    face = None
    try:
        face = data['face']['conf']
    except: 
        pass
    
    if face is not None and face > 0.95:
        cv2.imwrite('final_data/front/' + name + '.jpg', image)
    else:
        cv2.imwrite('final_data/back/' + name + '.jpg', image)


if __name__ == '__main__':
    pool = Pool(12)
    output = list(tqdm(
        pool.imap(crop, all_files), total=len(all_files), desc="Cropping"))
    pool.terminate()