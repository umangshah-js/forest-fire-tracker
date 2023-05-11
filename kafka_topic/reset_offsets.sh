cd kafka_2.13-3.4.0;
bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --reset-offsets --to-latest --all-groups --all-topics --execute