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

for file in all_files:
    if os.path.basename(file)[:5] == '0_fro':
        os.remove(file)