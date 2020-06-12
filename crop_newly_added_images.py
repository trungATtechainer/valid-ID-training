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
for i in range(len(all_files)):
    if len(os.path.basename(all_files[i])) < 14:
        all_files[i] = None

count = 0
for file in all_files:
    if file == None:
        count += 1

for i in range(count):
    all_files.remove(None)

def crop(file):
    image = cv2.imread(file)
    name = os.path.basename(file).split('.')[0]

    with open("data/json/{}.json".format(name[13:])) as json_file:
        data = json.load(json_file)
        shape = np.int32([data['crop_rotated_page']['polys']])[0]

        ## (1) Crop the bounding rect
    height, width, _ = image.shape
    a_y = height * random.uniform(0.05, 0.25)
    a_x = width * random.uniform(0.05, 0.25)

    rect = cv2.boundingRect(shape)
    x,y,w,h = rect

    min_x = int(max(x - a_x, 0))
    max_x = int(min(x + w + a_x, width))
    min_y = int(max(y - a_y, 0))
    max_y = int(min(y + h + a_y, height))

    croped = image[min_y:max_y, min_x:max_x].copy()
    try:
        cv2.imwrite('data/1_final/' + name[13:] + '.jpg', croped)
    except:
        print(name)

if __name__ == '__main__':
    pool = Pool(12)
    output = list(tqdm(
        pool.imap(crop, all_files), total=len(all_files), desc="Cropping"))
    pool.terminate()