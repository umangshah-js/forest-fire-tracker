import os
import glob
from utils import *
from tqdm.auto import tqdm
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np

# import multiprocessing as mp
# Pool = mp.Pool(mp.cpu_count())


# Hyperparameters
data_pth = "/home/mohit/NYU/Big_Data/Homework/final_proj/data"
m,n = 24,24
x_dim, y_dim = 256, 256
timestamp = 10
location = m*n # 24*24 locations


states = {
    0: "empty",
    1: "alive",
    2: "heating",
    3: "burning",
    4: "dead"
}


contours = []
for i in tqdm(range(0, location),desc="Init Contours"):
    x = (i//24)
    y = (i%24)

    img_path = f"{data_pth}/raw/Cam_{x}_{y}"
    init_contour_path = f"{data_pth}/init_contour/"
    os.makedirs(init_contour_path, exist_ok=True)

    im = cv2.imread(img_path+"/0.png")
    
    tmp_contour = get_contour(im, radius=4, save_path = init_contour_path+f"{x}_{y}.npy")
    contours.append(tmp_contour)


for j in tqdm(range(0,timestamp),desc="Timestamp"):
    time_stmp = np.zeros((m * x_dim, n * y_dim), dtype=np.uint8)
    
    time_arr_save = f"{data_pth}/timestamps/arr"
    os.makedirs(time_arr_save, exist_ok=True)

    time_im_save = f"{data_pth}/timestamps/images"
    os.makedirs(time_im_save, exist_ok=True)

    for i in range(0, location):    
        x = (i//24)
        y = (i%24)

        # os.makedirs(f"{data_pth}/contours/{x}_{y}", exist_ok=True)
        img_path = f"{data_pth}/raw/Cam_{x}_{y}"
        tmp_contour = contours[i]
     
        im = cv2.imread(img_path+f"/{j}.png")

        contour = get_color_contour(im, tmp_contour)
        
        time_stmp[(m-1-x)*x_dim:(m-x)*x_dim, y*y_dim: (y+1)*y_dim] = contour

    time_stmp_im = arr_to_image(time_stmp)

    # Saving the timestamp image and array
    cv2.imwrite(f"{time_im_save}/{j}.png", time_stmp_im)
    np.save(f"{time_arr_save}/{j}.npy", time_stmp)
   