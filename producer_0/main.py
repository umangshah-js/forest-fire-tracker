from kafka import KafkaProducer, KafkaConsumer, TopicPartition
import glob
import json
import redis
import time

r = redis.Redis(host="localhost", port=6379, db=0)


def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)


def on_send_error(excp):
    log.error("I am an errback", exc_info=excp)


event_consumer = KafkaConsumer(
    group_id="image_producer", bootstrap_servers=["localhost:9092"]
)

event_consumer.assign([TopicPartition("wfs-events", 0)])
event_consumer.commit()


def main():
    producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
    topic_name = "image-preprocess"
    data_location = None
    with open("../env.json") as f:
        env = json.load(f)
        print(env)
        data_location = env["data_path"]
    if data_location == None:
        print("Path not set")
        return 1

    timestamp = 0
    for timestamp in range(150):
        for img_file in glob.glob(f"{data_location}/raw/Cam_*/{str(timestamp)}.png"):
            key = f"""{img_file.split("/")[-2]}_{str(timestamp)}"""
            with open(img_file, "rb") as img:
                val = img.read()
                r.set(key, val)
                producer.send(
                    topic_name, key=key.encode(), value=b"contour detection"
                ).add_callback(on_send_success).add_errback(on_send_error)
        producer.flush()
        # New timestamp. wait for event consumer to publish event
        for event_message in event_consumer:
            print(event_message.key, event_message.value)
            if event_message.key.decode() == "time_stamp":
                print(
                    f"Received signal to process next timestamp {event_message.value.decode()}"
                )
                curr_timestamp = int(event_message.value.decode())
                event_consumer.commit()
                break


main()
