from kafka import KafkaProducer
import glob
def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)

def on_send_error(excp):
    log.error('I am an errback', exc_info=excp)


def main():
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    topic_name = "image-preprocess-1"
    
    # todo: Get path as args
    data_location = "/media/umang/Windows 10/Big Data/llcpp/Wildfire_sim_Data/2023-05-04-23-18-11"

    timestamp = 0
    for timestamp in range(100):
        for img_file in glob.glob(f"{data_location}/*/{str(timestamp)}.png"):
            key = f"""{img_file.split("/")[-2]}_{str(timestamp)}""".encode()
            with open(img_file, 'rb') as img:
                val = img.read()
                producer.send(topic_name, key=key, value=val).add_callback(on_send_success).add_errback(on_send_error)
        producer.flush()

main()