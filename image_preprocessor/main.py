from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import numpy as np
import os
import sys
import redis
from cv_utils import *
import msgpack
import msgpack_numpy as mnp
import argparse

parser = argparse.ArgumentParser(
    prog="Image Preprocessor",
    description="Detects centers and their states",
    epilog="provide instance_id",
)

parser.add_argument("id")
args = parser.parse_args()
instance_id = args.id


# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(
    "image-preprocess",
    group_id="image_preprocess",
    bootstrap_servers=["localhost:9092"],
)

producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
producer_topic_name = "fire-prediction"

# event_consumer = KafkaConsumer(
#                          group_id=f"image_preprocess_1_{str(instance_id)}",
#                          bootstrap_servers=['localhost:9092'])
# event_consumer.assign([TopicPartition('wfs-events', 0)])
# event_consumer.commit()


def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)


def on_send_error(excp):
    log.error("I am an errback", exc_info=excp)


r = redis.Redis(host="localhost", port=6379, db=0)


curr_timestamp = 0
m, n = 24, 24
x_dim, y_dim = 256, 256
print("Consuming messages")
processed = 0
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    # print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
    #                                       message.offset, message.key,
    #                                       message.value))
    print(message.key.decode())
    [_, x, y, time_stamp] = message.key.decode().split("_")
    x = int(x)
    y = int(y)
    time_stamp = int(time_stamp)
    print(x, y, time_stamp)
    centers = None
    # print("Fetched from redis: ",r.get(message.key.decode()))
    image_np = np.frombuffer(r.get(message.key.decode()), np.uint8)
    r.delete(message.key.decode())
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    redis_key = "_".join(message.key.decode().split("_")[:-1]) + "_centers"
    if time_stamp == 0:
        arr, centers = get_contour(img, radius=10, save_path=None)
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
    print(processed)

    # print(message.key.decode(),len(centers))
    # break
