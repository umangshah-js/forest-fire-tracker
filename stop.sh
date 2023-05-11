for pid in $(pgrep -f "python main.py"); do kill $pid; done
for pid in $(pgrep -f "python img_proc/main.py"); do kill $pid; done
for pid in $(pgrep -f "kafka_exporter"); do kill $pid; done
# cd kafka_topic
# bash -x stop.sh
cd ..
rm -rf "/media/umang/Windows 10/Big Data/llcpp/Wildfire_sim_Data/data/final_images/*"
