FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && \
    apt-get install -y \
    libegl-dev \
    cmake \
    git \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt
