FROM python:3.8-slim-buster
RUN pip install kafka-python
RUN pip install redis
WORKDIR /usr/src/app
COPY main.py main.py
CMD ["python", "main.py"]