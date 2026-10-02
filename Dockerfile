# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - Participant demo application (Docker submission template)
# ---------------------------------------------------------------------------
# The participant application must be developed/submitted as a Docker image (required).
# The standard entry point is `docker compose up`, and the team ID/token are
# provided at runtime through .env or environment variables (see .env.example).
#
# Build context = this repo root (copies marc_sdk + demo together).
#   -> see compose build.context=. , dockerfile=Dockerfile.
# ---------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 + ROS2 Humble + Python 3.10 (provides rclpy / *_msgs, default RMW=FastDDS)
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Participant algorithm dependencies. numpy + pyyaml (stdlib/ROS2 for the rest), plus placo
# for the closed-loop arm IK pick (Pinocchio-based, pip-installable on Python 3.10), plus
# llama-cpp-python for the offline NLU parser (see nlu/parser.py), plus torch/transformers/
# ultralytics for the detection stage (detection/detector.py's YOLODetector + OwlV2Detector,
# combined in HybridDetector -- see detection/grounding.py's _default_detector).
# The network is used at build time only -- the judging runtime environment has no internet (no runtime dependency).
#
# 2026-08-27: torch/transformers/ultralytics and both committed YOLO checkpoints are used:
# the original model for full CCTV frames and the crop-trained 10-target model for the
# landmark-centred second look. `COPY detection` below includes both weights. If the ROI
# checkpoint is unavailable, grounding logs the failure and safely keeps the full-frame path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-yaml python3-pip curl \
    && pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir placo llama-cpp-python torch transformers ultralytics \
    && rm -rf /var/lib/apt/lists/*

# manipulation/grasp.py imports scipy.spatial.transform.Rotation; split into its own layer
# (rather than appended to the line above) so this addition doesn't bust the ~2GB+ pip-install
# layer's build cache -- torch/transformers/ultralytics don't need to redownload for this.
RUN pip3 install --no-cache-dir scipy

WORKDIR /agent

# --- Everything below that hits the network goes BEFORE the source COPY, so editing a
#     .py file never busts the (slow, ~2.4 GB) model-download cache below.

# OpenCV (pulled in by ultralytics) dlopen()s libGL / libglib at import time, and the
# ros:humble-ros-base image ships neither. Without them `from ultralytics import YOLO`
# raises "libGL.so.1: cannot open shared object file", the YOLO path never loads, and
# HybridDetector silently falls back to OWLv2 alone.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# OWLv2 weights: baked in at build time (internet allowed then, see NOTICES.md) so the
# judging runtime -- which has no internet -- never has to call out on first use. Same
# model id as detection/detector.py's OwlV2Detector default. HF_HUB_OFFLINE/
# TRANSFORMERS_OFFLINE force every later huggingface_hub/transformers call to use this
# cache only, so a cache-miss fails loudly instead of silently trying the network.
# Needs only torch+transformers (installed above), no /agent source -- hence it sits here.
RUN python3 -c "\
from transformers import Owlv2Processor, Owlv2ForObjectDetection; \
Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble'); \
Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# NLU model weights: too large for git, so fetched here at build time (internet allowed only
# at build time -- see FAQ notice on runtime internet being blocked). Same file used in local
# dev (nlu/test_parser.py); keep this URL in sync with the one used to fetch it there.
# Using 3B, not 1.5B: after target_type became pose-qualified (9 person labels instead of
# plain "person", 2026-08-20), a re-run of the size comparison on the harder task showed 3B
# leads 1.5B by +8pp on target_type (86% vs 78%) -- see nlu/README.md "Qwen2.5-3B가 최선의
# 선택인가". 3B's cold-start latency spike (~10-18s, first inference call only) is absorbed by
# the warmup call in nlu/parser.py's NLUParser.__init__, well within the confirmed 30s Stage1
# time_limit (steady-state latency after warmup is ~0.6-1.0s/call).
# --retry/--speed-limit: this ~1.8GB download can stall mid-transfer on a flaky connection
# (hit this locally more than once) -- without retry flags that kills the whole build.
# Written into /agent/nlu/models/ before `COPY nlu` below; COPY merges into that dir and
# leaves this file in place (the host's nlu/models/*.gguf is .dockerignore'd anyway).
RUN mkdir -p /agent/nlu/models && curl -L --retry 5 --retry-delay 5 --connect-timeout 15 \
        --speed-limit 1000 --speed-time 30 \
        -o /agent/nlu/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
        "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# context = repo root -> copy marc_sdk (library) + demo (agent) + the three pipeline
# stages, one per teammate: nlu (language) / detection (vision) / geometry (coordinates).
# Anything not COPYed here simply does not exist in the image -- a missing stage shows up
# as ModuleNotFoundError at scoring time, not at build time.
# `tools/` is intentionally absent: dev-only recorder/verifier, never used by the agent.
COPY marc_sdk  /agent/marc_sdk
COPY demo      /agent/demo
COPY nlu       /agent/nlu
COPY detection /agent/detection
COPY geometry  /agent/geometry
COPY manipulation /agent/manipulation

# Same RMW as the runtime platform (FastDDS). Import paths for marc_sdk (/agent) + demo local modules (/agent/demo).
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PYTHONPATH=/agent:/agent/demo

WORKDIR /agent/demo

# The ros:humble ENTRYPOINT (/ros_entrypoint.sh) sources /opt/ros/humble/setup.bash
# and then runs the CMD below. team_id/token are injected via the compose env (from_env).
CMD ["python3", "participant_app.py"]
