FROM ros:humble

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system build tools, git, and ROS tools
# Added: ros-humble-foxglove-bridge
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    build-essential \
    ros-humble-foxglove-bridge \
    ros-humble-moveit \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-moveit-planners \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Workspace & Clone ROS-TCP-Endpoint
WORKDIR /ros_ws
RUN mkdir -p src
WORKDIR /ros_ws/src

RUN for i in 1 2 3 4 5; do \
        git clone -b main-ros2 https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git && break || \
        (echo "Clone attempt $i failed, retrying in 10s..." && sleep 10); \
    done

# Patch ROS-TCP-Endpoint bug: send_topic_list crashes with AttributeError when
# any registered node has msg=None (not yet initialized at topic-list request time).
# Guard the list comprehension so None-msg nodes are silently skipped.
RUN sed -i \
    's/else self\.parse_message_name(node\.msg)/else (self.parse_message_name(node.msg) if node is not None and node.msg is not None else "")/' \
    /ros_ws/src/ROS-TCP-Endpoint/ros_tcp_endpoint/tcp_sender.py

# 3. Copy AR4 URDF and meshes
COPY ar4_moveit_config /ros_ws/src/ar4_moveit_config

# 4. Install Dependencies via rosdep
WORKDIR /ros_ws
RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# 5. Build the Workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# 6. Source setup in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> /root/.bashrc

# Expose ports: 10000 (Unity), 8765 (Foxglove), 5020 (Python Bridge)
EXPOSE 10000
EXPOSE 8765
EXPOSE 5020

# 7. Default Command (kept as Unity endpoint)
CMD ["/bin/bash", "-c", \
    "source /opt/ros/humble/setup.bash && \
     source /ros_ws/install/setup.bash && \
     ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000"]