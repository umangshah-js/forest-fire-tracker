set -m
cd kafka_topic;
# rm -rf kafka_2.13-3.4.0/logs/*kafka*
sleep 2
bash -x setup.sh;
wait
bash -x reset_offsets.sh
wait
cd ..
cd kafka_exporter
bash -x run.sh > kafka_exporter.log 2>&1 &
cd ..
redis-cli flushdb
cd dashboard
python main.py > dashboard.logs 2>&1 &
sleep 5
google-chrome http://127.0.0.1:8050 --force-device-scale-factor=1 --start-maximized > /dev/null 2>&1 &
cd ..
python img_proc/main.py > prediction_engine.log 2>&1 &
sleep 2
cd image_preprocessor
bash -x run.sh
cd ..
cd producer_0
python main.py > producer_0.log 2>&1 &
cd ..
tail -f prediction_engine.log


