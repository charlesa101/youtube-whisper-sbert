
ENV := dev
CLUSTER_NAME := youtube-whisper-sbert
GCP_PROJECT := youtube-whisper-sbert
REGION := us-central1-a
SHELL := /bin/zsh
CWD:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq ($(ENV), dev)
	SKAFFOLD_DEFAULT_REPO := "us-docker.pkg.dev/youtube-whisper-sbert/containers"
endif

.PHONY: skaffold-build
skaffold-build:
	gcloud auth configure-docker us-docker.pkg.dev
	echo 'building container for: $(SKAFFOLD_DEFAULT_REPO)'
	skaffold build  --platform linux/amd64 --default-repo=$(SKAFFOLD_DEFAULT_REPO) --push