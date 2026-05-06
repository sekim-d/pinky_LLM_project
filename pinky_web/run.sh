#!/bin/bash
source /opt/ros/jazzy/setup.bash
source ~/pinky_LLM_project/pinky/install/setup.bash
source /home/seongeun/venv/jazzy/bin/activate
cd /home/seongeun/pinky_LLM_project/pinky_web
python3 app.py
