from dask.distributed import Client
import numpy as np
import dask.array as da
import cv2
from utils import *
from tqdm.auto import tqdm
import os
import h5py


def prediction(
    npzfile_pth,
    contour_chunk=(256, 256),
    center_chunk=(4000, 4),
    eps=1e-3,
    k1=0.001,
    k2=10,
    alpha=5,
    offset=7,
    dim=6155,
    img_save_path="test.png",
    zarr_save_path="centers.zarr",
    npy_save_path="img.npy",
    hdf5_save_path="centers.hdf5",
    f=open("prediction.log", "a+"),
):
    # countours_da  = da.from_array(npzfile['contours'], chunks=(512, 512))
    # centers_da = da.from_array(npzfile['centers'], chunks=(512, 4))

    states = {
        0: np.array([255, 255, 255], dtype=np.uint8),  # nothing
        1: np.array([0, 255, 0], dtype=np.uint8),  # "alive",
        2: np.array([0, 255, 255], dtype=np.uint8),  # "heating",
        3: np.array([0, 0, 255], dtype=np.uint8),  # "burning",
        4: np.array([0, 0, 0], dtype=np.uint8),  # "dead",
        5: np.array([255, 0, 0], dtype=np.uint8),  # "fireline",
    }

    # print(f"Overall number of trees: {centers_da.shape[0]}")

    # countours = da.from_array(np.load(npzfile_pth)['contours'], chunks=contour_chunk)
    centers = da.from_array(np.load(npzfile_pth)["centers"], chunks=center_chunk)

    # Get the burning centers
    ind = (centers[:, 3] >= 2) & (centers[:, 3] < 4) & (centers[:, 2] >= 10)
    burning_centers = centers[ind, :]
    # heat = da.from_array(np.zeros((centers.shape[0], 1)),chunks=(center_chunk[0],1))

    # Computing Chunk sizes
    # heat.compute_chunk_sizes()
    # burning_centers.compute_chunk_sizes()
    # print(f"Number of burning trees: {burning_centers.shape[0]}", file=f)

    # Get the heats for each center
    # for i in range(burning_centers.shape[0]):
    #     norm = da.linalg.norm(centers[:,:2] - burning_centers[i, :2], axis=1, ord=2)[...,None]
    #     heat += (k2/(norm+eps) + k1/(1+(da.exp(((norm/10000)+eps)))))*100
    coord = centers[:, :2][..., None]
    diff = coord - burning_centers[:, :2].T
    norm = da.linalg.norm(diff, axis=1, ord=2)
    norm = k2 / (norm + eps) + k1 / (1 + (da.exp(((norm / 10000) + eps))))
    heat = da.sum(norm, axis=1)[..., None]

    # Setting the fireline
    # Using mean and std deviation to set the threshold

    lim = heat[~ind].mean()
    lim_std = heat[~ind].std()

    # plt.hist(tmp,bins=100)
    # plt.xlabel("Heat")
    # plt.ylabel("Frequency")
    # plt.title("Heat Distribution")
    # plt.show()
    # print(f"Trees ommited in distribution: {da.sum(ind).compute()}", file = f)

    # print(f"Limiting heat value: {lim} with std deviation: {lim_std}", file = f)

    thresh_max = lim + alpha * lim_std + offset * lim_std
    thresh_min = lim - alpha * lim_std + offset * lim_std
    # print(f'thresh_min: {thresh_min}, thresh_max: {thresh_max}', file = f)

    # Setting the alive trees on heating
    ind2 = heat >= thresh_min
    ind_heat = ind2[:, 0] & (centers[:, 3] == 1)
    centers[ind_heat, 3] = 5  # fireline

    # print(f"""
    # Number of trees alive(Green): {centers[centers[:,3]==1].shape[0]}
    # Number of trees on fireline(Blue): {centers[centers[:,3]==5].shape[0]}
    # Number of trees heating(Yellow): {centers[centers[:,3]==2].shape[0]}
    # Number of trees burning(Red): {centers[centers[:,3]==3].shape[0]}
    # Number of trees dead(Black): {centers[centers[:,3]==4].shape[0]}
    # """, file = f)

    # Store the centers array in zarr format
    centers.to_zarr(zarr_save_path, overwrite=True)
    # da.to_hdf5(hdf5_save_path, "/centers", centers)

    # For visualization and testing purposes only
    img = np.full((int(dim), int(dim), 3), 255, dtype=np.uint8)
    img = draw_circle(img=img, centers=centers.compute(), states=states, thickness=-1)
    cv2.imwrite(img_save_path, img)
    np.save(npy_save_path, img)

    del (
        centers,
        heat,
        burning_centers,
        ind,
        ind2,
        ind_heat,
        thresh_max,
        thresh_min,
        lim,
        lim_std,
        coord,
        diff,
        norm,
    )


if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")
    print(client)
    timestamps = 153
    f = open("prediction.log", "w+")

    for i in tqdm(range(0, timestamps)):
        npzfile_pth = f"/home/mohit/NYU/Big_Data/Homework/forest-fire-tracker/data/timestamps/arr/{i}.npz"
        print(f"{'-'*20} For timestamp {i} {'-'*20}", file=f)

        img_save_path = (
            f"/home/mohit/NYU/Big_Data/Homework/forest-fire-tracker/data/prediction/img"
        )
        zarr_save_path = f"/home/mohit/NYU/Big_Data/Homework/forest-fire-tracker/data/prediction/zarr"

        npy_save_path = f"/home/mohit/NYU/Big_Data/Homework/forest-fire-tracker/data/prediction/npy/"

        hdf5_save_path = f"/home/mohit/NYU/Big_Data/Homework/forest-fire-tracker/data/prediction/hdf5"

        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(zarr_save_path, exist_ok=True)
        os.makedirs(npy_save_path, exist_ok=True)
        os.makedirs(hdf5_save_path, exist_ok=True)

        prediction(
            npzfile_pth,
            img_save_path=f"{img_save_path}/{i}.png",
            zarr_save_path=f"{zarr_save_path}/{i}.zarr",
            npy_save_path=f"{npy_save_path}/{i}.npy",
            hdf5_save_path=f"{hdf5_save_path}/{i}.hdf5",
            f=f,
        )
        print(f"Done with {i}.png", file=f)
        # break

    f.close()
    client.close()
