FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/bridge
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/opt/bridge/src
CMD ["python3", "-m", "bridge.smoke_test"]
