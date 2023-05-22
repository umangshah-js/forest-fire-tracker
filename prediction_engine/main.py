from dask.distributed import Client
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from matplotlib import pyplot as plt
from utils import *
from tqdm.auto import tqdm
import os
import json
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import msgpack
import msgpack_numpy as mnp
import time
# import cupy
import redis
# def load_config(config_path):
#     config = None
#     with open(config_path, "r") as f:
#         config = json.load(f)
#     print(config)
#     if config["gpu"]:
#         import cupy

#     return config
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
SIMULATION_DATA_PATH = os.getenv("SIMULATION_DATA_PATH", None)
def prediction(
    # npzfile_pth,
    centers,
    contour_chunk=(256, 256),
    center_chunk=(5000, 4),
    eps=1e-3,
    k1=0.001,
    k2=10,
    alpha=5,
    offset=7,
    dim=6155,
    img_save_path="test.png",
    zarr_save_path="centers.zarr",
    # f=open("prediction.log", "a+"),
    stats_file=None,
):
    # countours_da  = da.from_array(npzfile['contours'], chunks=(512, 512))
    # centers_da = da.from_array(npzfile['centers'], chunks=(512, 4))
    # print(centers)
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
    # centers = da.from_array(np.load(npzfile_pth)["centers"], chunks=center_chunk)

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

    centers = centers.compute()
    if(stats_file!=None):
        json.dump({
            "alive":centers[centers[:,3]==1].shape[0]+centers[centers[:,3]==5].shape[0],
            "predicted_alive":centers[centers[:,3]==1].shape[0],
            "fireline": centers[centers[:,3]==5].shape[0],
            "heating": centers[centers[:,3]==2].shape[0],
            "burning": centers[centers[:,3]==3].shape[0],
            "dead": centers[centers[:,3]==4].shape[0],
        },stats_file)
        stats_file.close()

    # print(
    #     f"""
    # Number of trees alive(Green): {centers[centers[:,3]==1].shape[0]}
    # Number of trees on fireline(Blue): {centers[centers[:,3]==5].shape[0]}
    # Number of trees heating(Yellow): {centers[centers[:,3]==2].shape[0]}
    # Number of trees burning(Red): {centers[centers[:,3]==3].shape[0]}
    # Number of trees dead(Black): {centers[centers[:,3]==4].shape[0]}
    # """,
    #     file=f,
    # )

    # Store the centers array in zarr format
    # centers.to_zarr(zarr_save_path, overwrite=True)

    lst = zarr_save_path.split(".")[0]
    np.save(lst + ".npz", centers)
    # For visualization and testing purposes only
    img = np.full((int(dim), int(dim), 3), 255, dtype=np.uint8)
    cv2.imwrite(
        img_save_path,
        draw_circle(img=img, centers=centers, states=states, thickness=-1),
    )

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


def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)


def on_send_error(excp):
    print("I am an errback", exc_info=excp)

consumer = None
def connect_to_kafka():
    global consumer
    try:
        consumer = KafkaConsumer(
            group_id="prediction_engine",
            bootstrap_servers=[KAFKA_BROKER],
        )
        consumer.assign([TopicPartition("fire-prediction",0)])
        consumer.seek_to_beginning()
        consumer.commit()
    except Exception as e:
        time.sleep(1)
        print("retrying connecting to kafka")
        connect_to_kafka()

if __name__ == "__main__":
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="500MB")
    # cluster = LocalCUDACluster(n_workers=1, threads_per_worker=100) 
    # client = Client(cluster)
    print(client)
    # time_stamp = 0
    num_chunks_total = 24 * 24
    center_chunk = (5000, 4)
    connect_to_kafka()
    event_producer = KafkaProducer(bootstrap_servers=[KAFKA_BROKER])
    event_topic_name = "wfs-events"
    print("Publishing event",event_producer.send(
        event_topic_name,
        key=b"time_stamp",
        value=b"0",
        partition=0,
    ).add_errback(on_send_error))
    event_producer.flush()
    print("Event sent")
    config = None
    zarr_save_path = None
    img_save_path = None

    # config = load_config("env.json")

    data_path = SIMULATION_DATA_PATH
    gpu = False
    zarr_save_path = f"{data_path}/zarr"
    img_save_path = f"{data_path}/final_images"

    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(zarr_save_path, exist_ok=True)
    
    # with open("prediction.log", "w+") as log_file:
    for timestamp in tqdm(range(150)):
        centers_list = []

        num_chunks = 0
        print(f"Consuming messages for timestamp {str(timestamp)}", flush=True)
        for message in consumer:
            num_chunks += 1
            print(message.key.decode(),num_chunks,num_chunks_total, flush=True)
            centers_coloured_packed = message.value
            centers_coloured = msgpack.unpackb(
                centers_coloured_packed, object_hook=mnp.decode
            )
            
            
            if gpu:
                cpy_centers = cupy.array(centers_coloured)
                da_centers = da.from_array(centers_coloured,asarray=False)
                # da_centers = da_centers.map_blocks(cupy.asarray)
            else:   
                da_centers = da.from_array(centers_coloured)
            centers_list.append(da_centers)

            # print(centers.flags.writeable)
            if num_chunks == num_chunks_total:
                # print("Received all data for the timestamp")
                event_producer.send(
                    event_topic_name,
                    key=b"time_stamp",
                    value=str(timestamp + 1).encode(),
                    partition=0,
                ).add_errback(on_send_error)
                event_producer.flush()
                break
        consumer.commit()
        centers = da.concatenate(centers_list)
        centers = da.rechunk(centers, chunks=center_chunk)
        start = time.time()
        stats_file = open(f"{img_save_path}/{timestamp}.json", "w+")
        prediction(
            centers,
            zarr_save_path=f"{zarr_save_path}/{timestamp}.zarr",
            img_save_path=f"{img_save_path}/{timestamp}.png",
            stats_file=stats_file,
        )
        print("Time to predict: ", time.time() - start)
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()
    
    
