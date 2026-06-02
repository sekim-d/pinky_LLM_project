FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install fastapi uvicorn pydantic --break-system-packages

WORKDIR /ros2_ws/src
COPY msgs ./msgs
COPY task_manager ./task_manager

WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select msgs task_manager

WORKDIR /app
CMD ["/bin/bash", "-c", \
     "source /opt/ros/jazzy/setup.bash && \
      source /ros2_ws/install/setup.bash && \
      ros2 run task_manager task_manager"]
