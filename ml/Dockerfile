# Render/Railway Dockerfile for ML Server
FROM python:3.11-slim

WORKDIR /app

# Install git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (including .git for LFS)
COPY . .

# Pull LFS files (requires .git directory to be present)
# Note: For Docker builds, LFS files should be pulled before docker build
# or use multi-stage build with git clone

EXPOSE 8080

CMD ["gunicorn", "server.app:app", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "2"]
