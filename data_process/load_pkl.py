import pickle

pkl_path = "/home/wangxinqun/code/simple-face-det/data/train000.pkl"
with open(pkl_path, "rb") as f:
    pkl_content = pickle.load(f)

print("end")