from matplotlib import pyplot as plt
import cv2
from utils import *
from tqdm.auto import tqdm
import os
import tracemalloc
import numpy as np

def prediction(npzfile,
                eps = 1e-3,
                k1 = 0.00001,
                k2 = 10,
                alpha = 5,
                offset = 7,
                save_path = "test.png",
                f = open("prediction.log","a+")
            ):
    # countours_da  = da.from_array(npzfile['contours'], chunks=(512, 512))
    # centers_da = da.from_array(npzfile['centers'], chunks=(512, 4))

    states = {
        0: np.array([255,255,255], dtype=np.uint8), #nothing
        1: np.array([0,255,0], dtype=np.uint8), # "alive",
        2: np.array([0,255,255], dtype=np.uint8), # "heating",
        3: np.array([0,0,255], dtype=np.uint8), # "burning",
        4: np.array([0,0,0], dtype=np.uint8), # "dead",
        5: np.array([255,0,0], dtype=np.uint8) # "fireline",
    }


    # print(f"Overall number of trees: {centers_da.shape[0]}")



    countours = npzfile['contours']
    centers = npzfile['centers']

    # Get the burning centers
    ind = (centers[:, 3]>=2) & (centers[:,3]<4) & (centers[:, 2]>=10)
    burning_centers = centers[ind,:]
    heat = np.zeros((centers.shape[0], 1))

    print(f"Number of burning trees: {burning_centers.shape[0]}", file=f)


    # Get the heats for each center

    for i in range(burning_centers.shape[0]):
        norm = np.linalg.norm(centers[:,:2] - burning_centers[i, :2], axis=1, ord=2)[...,None]
        heat += (k2/(norm+eps) + k1/(1+(np.exp(((norm/10000)+eps)))))*100

    # Setting the fireline
    # Using mean and std deviation to set the threshold
    tmp = heat.copy()
    tmp[ind]=0
    lim = tmp.mean()
    lim_std = tmp.std()

    # plt.hist(tmp,bins=100)
    # plt.xlabel("Heat")
    # plt.ylabel("Frequency")
    # plt.title("Heat Distribution")
    # plt.show()
    print(f"Trees ommited in distribution: {sum(ind)}", file = f)


    print(f"Limiting heat value: {lim} with std deviation: {lim_std}", file = f)

    thresh_max = lim + alpha*lim_std + offset*lim_std
    thresh_min = lim - alpha*lim_std + offset*lim_std
    print(f'thresh_min: {thresh_min}, thresh_max: {thresh_max}', file = f)

    # Setting the alive trees on heating
    ind2 = (heat>=thresh_min) 
    ind_heat = ind2[:,0] & (centers[:,3]==1)
    centers[ind_heat, 3] = 5    # fireline

    print(f"""
    Number of trees alive(Green): {centers[centers[:,3]==1].shape[0]}
    Number of trees on fireline(Blue): {centers[centers[:,3]==5].shape[0]}
    Number of trees heating(Yellow): {centers[centers[:,3]==2].shape[0]}
    Number of trees burning(Red): {centers[centers[:,3]==3].shape[0]}
    Number of trees dead(Black): {centers[centers[:,3]==4].shape[0]}
    """, file = f)

    dim = max(np.max(centers[:,1]), np.max(centers[:,0])) +1
    img = np.full((int(dim), int(dim), 3), 255, dtype=np.uint8)

    cv2.imwrite(save_path, draw_circle(img=img, centers=centers, states=states, thickness=-1) )

    return tracemalloc.get_traced_memory()


if __name__=="__main__":
    tracemalloc.start()
    memory_usage = []
    memory_peak = 0
    timestamps = 153
    f = open("prediction.log","w+")
    
    for i in tqdm(range(timestamps)):
        
        npzfile = np.load(f"/home/mohit/NYU/Big_Data/Homework/forest-fire-tracker/data/timestamps/arr/{i}.npz")
        print(f"{'-'*20} For timestamp {i} {'-'*20}", file = f)
        os.makedirs(f"../data/prediction/", exist_ok=True)
        mem = prediction(npzfile, save_path=f"../data/prediction/{i}.png", f = f)
        print(f"Done with {i}.png", file = f)

        memory_usage.append(mem[0])
        memory_peak = max(memory_peak, mem[1])

        # break
    f.close()

    
    print(f'Peak memory usage was {memory_peak / 10**6}MB')
    plt.plot(range(len(memory_usage)),memory_usage, label="memory usage")
    plt.show()