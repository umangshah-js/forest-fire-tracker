From python:3.8-slim-buster
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install kafka-python redis opencv-python numpy msgpack_numpy msgpack 
RUN pip install "dask[complete]" tqdm matplotlib pandas numba
WORKDIR /usr/src/app
COPY main.py main.py
COPY utils.py utils.py
CMD ["python", "main.py"]