from kafka import KafkaProducer, KafkaConsumer, TopicPartition
import glob
import json
import redis
import time
import os
print("Start", flush=True)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
SIMULATION_DATA_PATH = os.getenv("SIMULATION_DATA_PATH", None)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)



def on_send_success(record_metadata):
    print(record_metadata.topic)
    print("partition",record_metadata.partition)
    print(record_metadata.offset)


def on_send_error(excp):
    log.error("I am an errback", exc_info=excp)
    
event_consumer = None
connected = False
print("Trying to connect")
while not connected:
    try:
        event_consumer = KafkaConsumer(
            group_id="image_producer", bootstrap_servers=[KAFKA_BROKER])
        connected = True
    except:
        print("Retying connecting to kafka", flush=True)
        time.sleep(1)
event_consumer.assign([TopicPartition("wfs-events", 0)])
event_consumer.seek_to_beginning()
print("Waiting to begin",flush=True)
for event_message in event_consumer:
    print(event_message.key, event_message.value,flush=True)
    if event_message.key.decode() == "time_stamp":
        print(
            f"Received signal to process next timestamp {event_message.value.decode()}"
        )
        curr_timestamp = int(event_message.value.decode())
        event_consumer.commit()
        break

time.sleep(10)
print("Beginning",flush=True)
def main():
    producer = KafkaProducer(bootstrap_servers=[KAFKA_BROKER])
    topic_name = "image-preprocess"
    data_location = SIMULATION_DATA_PATH
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
        for event_message in event_consumer:
            print(event_message.key, event_message.value)
            if event_message.key.decode() == "time_stamp":
                print(
                    f"Received signal to process next timestamp {event_message.value.decode()}"
                )
                curr_timestamp = int(event_message.value.decode())
                event_consumer.commit()
                break
        # New timestamp. wait for event consumer to publish event
        


main()
