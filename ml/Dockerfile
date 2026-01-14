# Render Dockerfile for ML Server
# Multi-stage build to properly fetch Git LFS files

# Stage 1: Clone repo and fetch LFS files
FROM alpine/git:latest AS lfs-fetch
RUN apk add --no-cache git-lfs
WORKDIR /repo
RUN git lfs install
# Clone the repo (shallow clone for speed)
ARG REPO_URL=https://github.com/caylanwilcox/Project-Genesis.git
RUN git clone --depth 1 ${REPO_URL} . && \
    git lfs pull

# Stage 2: Build the Python application
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from local context
COPY . .

# Copy LFS files from the git clone stage (overwrite pointers with real files)
COPY --from=lfs-fetch /repo/ml/v6_models/ ./v6_models/

EXPOSE 8080

CMD ["gunicorn", "server.app:app", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "2"]
