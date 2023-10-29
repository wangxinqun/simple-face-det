import pickle
import os
import numpy as np
from PIL import Image

def data_loader(dir):
    imgs, gts = [], []
    files = os.listdir(dir)
    for file in files:
        if file.endswith("jpg"):
            img_path = os.path.join(dir, file)
            pkl_file = img_path.replace("jpg", "pkl")
            with open(pkl_file, "rb") as pkl_f:
                pkl_content = pickle.load(pkl_f)
            gts.append(np.array(pkl_content))
            img = Image.open(img_path)
            img = img.resize((512, 512))
            imgs.append(np.array(img))
    return imgs, gts