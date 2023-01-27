FROM ubuntu:jammy

RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    ffmpeg \
    pkg-config \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*