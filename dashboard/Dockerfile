From python:3.8-slim-buster
RUN pip install redis dash flask-caching pandas dash-bootstrap-components prometheus-client seaborn requests
WORKDIR /usr/src/app
COPY main.py main.py
RUN mkdir assets
CMD ["python", "main.py"]