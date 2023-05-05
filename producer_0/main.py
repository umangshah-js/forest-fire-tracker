# from kafka import KafkaConsumer
# consumer = KafkaConsumer('wfs-events',
#                          group_id='producer_0',
#                          bootstrap_servers=['localhost:9092'])

# for message in consumer:
#     # message value and key are raw bytes -- decode if necessary!
#     # e.g., for unicode: `message.value.decode('utf-8')`
#     print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
#                                           message.offset, message.key,
#                                           message.value))


from kafka import KafkaProducer
import glob

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
topic_name = "wfs-events"
data_location = "/media/umang/Windows 10/Big Data/llcpp/Wildfire_sim_Data/2023-05-04-23-18-11"
def on_send_success(record_metadata):
    print(record_metadata.topic)
    print(record_metadata.partition)
    print(record_metadata.offset)

def on_send_error(excp):
    log.error('I am an errback', exc_info=excp)


timestamp = 0
for timestamp in range(100):
    for img_file in glob.glob(f"{data_location}/*/{str(timestamp)}.png"):
        key = f"""{img_file.split("/")[-2]}_{str(timestamp)}""".encode()
        with open(img_file, 'rb') as img:
            val = img.read()
            producer.send(topic_name, key=key, value=val).add_callback(on_send_success).add_errback(on_send_error)
    producer.flush()
# producer.send(topic_name, key=b'foo', value=b'bar').add_callback(on_send_success).add_errback(on_send_error)
