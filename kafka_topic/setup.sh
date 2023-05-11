cd kafka_2.13-3.4.0;
bin/zookeeper-server-start.sh -daemon config/zookeeper.properties ;
wait
bin/kafka-server-start.sh -daemon config/server.properties ;
# bin/kafka-server-start.sh config/server.properties ;
wait
bin/kafka-topics.sh --create --topic image-preprocess --bootstrap-server localhost:9092 --partitions 4 --if-not-exists
bin/kafka-topics.sh --create --topic wfs-events --bootstrap-server localhost:9092 --partitions 1 --if-not-exists
bin/kafka-topics.sh --create --topic fire-prediction --bootstrap-server localhost:9092 --partitions 1 --if-not-exists


# bin/kafka-topics.sh -bootstrap-server localhost:9092 --topic image-preprocess --delete 
# bin/kafka-topics.sh -bootstrap-server localhost:9092 --topic fire-prediction --delete 
# bin/kafka-topics.sh -bootstrap-server localhost:9092 --topic wfs-events --delete 
