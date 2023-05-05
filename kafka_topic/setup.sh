cd kafka_2.13-3.4.0;
bin/zookeeper-server-start.sh -daemon config/zookeeper.properties ;
bin/kafka-server-start.sh -daemon config/server.properties ;
wait
bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic wfs-events || bin/kafka-topics.sh --create --topic image-preprocess-0 --bootstrap-server localhost:9092

