# Dockerfile
# ==========
# PURPOSE:
#   Containerize the FastAPI inference server so it can be deployed anywhere.
#
# HOW IT WORKS:
#   Stage 1 — Base image
#       Use python:3.10-slim as base
#
#   Stage 2 — Install dependencies
#       Copy requirements.txt
#       Run pip install -r requirements.txt
#       Note: torch-geometric needs special install order (torch first)
#
#   Stage 3 — Copy project files
#       Copy src/ and api/ directories
#       Copy results/best_model.pt (trained model)
#       Copy data/processed/elliptic_graph.pt (graph data)
#
#   Stage 4 — Expose and run
#       EXPOSE 8000
#       CMD: uvicorn api.main:app --host 0.0.0.0 --port 8000
#
# TO BUILD:
#   docker build -t gnn-fraudnet .
#
# TO RUN:
#   docker run -p 8000:8000 gnn-fraudnet
#
# TODO: Write actual Dockerfile commands below

FROM python:3.10-slim

# TODO: Install dependencies, copy files, set entrypoint
