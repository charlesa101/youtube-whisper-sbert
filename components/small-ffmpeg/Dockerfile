FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install --no-install-recommends -y \
    ffmpeg && rm -rf /var/lib/apt/lists/*