import cv2
import numpy as np
import pickle
import os


def annotate_box(event, x, y, flag, para):
    imgH, imgW = para["img_shape"][0], para["img_shape"][1]
    if event == cv2.EVENT_LBUTTONDOWN:
        para["pts"].append([x/imgW, y/imgH])
    if event == cv2.EVENT_RBUTTONDOWN:
        para["pts"].append([x/imgW, y/imgH])
        np.array(para["pts"], dtype=float)
        print("annotation pts: ", para["pts"])
        save_pkl_name = para["img_name"].replace("jpg", "pkl")
        if not os.path.exists(save_pkl_name):
            pickle.dump(pts, open(save_pkl_name, "wb"))
            print(save_pkl_name, " is saved!")
        else:
            raise Exception("Annotation file exists!")
        return


data_path = "./data/"
for root, _, files in os.walk(data_path):
    for f in files:
        if f.endswith("jpg"):
            img_name = os.path.join(root, f)
            img = cv2.imread(img_name)
            pts = []
            cv2.namedWindow(img_name)
            cv2.setMouseCallback(img_name, annotate_box, {"pts": pts, "img_name": img_name, "img_shape": img.shape})
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
print("Finish Annotation")