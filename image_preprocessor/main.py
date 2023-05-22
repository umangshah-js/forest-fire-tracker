from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import numpy as np
import os
import sys
import redis
from cv_utils import *
import msgpack
import msgpack_numpy as mnp
import time
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
PARTITION = int(os.getenv("PARTITION","0"))
time.sleep(10)
# To consume latest messages and auto-commit offsets
connected = False
while not connected:
    try:
        consumer = KafkaConsumer(
            group_id="image_preprocess",
            bootstrap_servers=[KAFKA_BROKER],
        )
        consumer.assign([TopicPartition("image-preprocess",PARTITION)])
        print("Seeked to beginning", flush=True)
        connected = True
    except Exception as e:
        print(e)
        print("Retrying connecting to kafka",flush=True)
print("Connected",flush=True)
time.sleep(5)
producer = KafkaProducer(bootstrap_servers=[KAFKA_BROKER])
producer_topic_name = "fire-prediction"


def on_send_success(record_metadata):
    print(record_metadata.topic)
    print("partition",record_metadata.partition)
    print(record_metadata.offset)


def on_send_error(excp):
    print("I am an errback", exc_info=excp)


r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


curr_timestamp = 0
m, n = 24, 24
x_dim, y_dim = 256, 256
processed = 0
print("Consuming messages",flush=True)
for message in consumer:
    print(message.key.decode(),flush=True)
    [_, x, y, time_stamp] = message.key.decode().split("_")
    x = int(x)
    y = int(y)
    time_stamp = int(time_stamp)
    print(x, y, time_stamp)
    centers = None
    image_np = np.frombuffer(r.get(message.key.decode()), np.uint8)
    r.delete(message.key.decode())
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    redis_key = "_".join(message.key.decode().split("_")[:-1]) + "_centers"
    if time_stamp == 0:
        arr, centers = get_contour(img, radius=5, save_path=None)
        centers_packed = msgpack.packb(centers, default=mnp.encode)
        r.set(redis_key, centers_packed)
    else:
        centers_packed = r.get(redis_key)
        centers = msgpack.unpackb(centers_packed, object_hook=mnp.decode)

    if len(centers.shape) == 2:
        center_color = get_color_centers(img, centers)
        # Origin shift the centers
        center_color[:, 1] = (m - x) * x_dim - center_color[:, 1]
        center_color[:, 0] = y * y_dim + center_color[:, 0]
    else:
        center_color = np.empty((0, 4))
    centers_color_packed = msgpack.packb(center_color, default=mnp.encode)
    key = message.key.decode() + "_colored"
    # r.set(message.key.decode(), centers_color_packed)
    processed += 1
    producer.send(
        producer_topic_name, key=key.encode(), value=centers_color_packed
    ).add_callback(on_send_success).add_errback(on_send_error)
    producer.flush()
    print(processed,flush=True)

    # print(message.key.decode(),len(centers))
    # break
