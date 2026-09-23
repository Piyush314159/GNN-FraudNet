# Dockerfile
# ==========
# PURPOSE:
#   Containerize the FastAPI inference server so it can be deployed anywhere.
#
# STRATEGY:
#   Download large wheels (torch) on the HOST first using pip download,
#   then COPY them into the image to avoid Docker's slow virtual network.
#
# TO BUILD:
#   1. pip download torch==2.4.0 --platform manylinux2014_aarch64 --python-version 311 --only-binary=:all: -d ./docker_wheels
#   2. docker build -t gnn-fraudnet .
#
# TO RUN:
#   docker run -p 8000:8000 gnn-fraudnet

FROM python:3.11-slim

WORKDIR /app

# Generous timeout for remaining (smaller) packages
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# 1) Install torch from pre-downloaded local wheels (avoids Docker network timeout)
COPY docker_wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels/ torch==2.4.0 && \
    rm -rf /tmp/wheels/

# 2) Install remaining inference-only deps (all small packages, no torchvision/scipy/etc.)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 3) Copy only what the API needs
COPY src/ ./src/
COPY api/ ./api/
COPY results/best_graphsage.pt ./results/best_graphsage.pt
COPY data/processed/elliptic_graph.pt ./data/processed/elliptic_graph.pt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
