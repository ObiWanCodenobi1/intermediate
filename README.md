# ICUAS 2026 UAV Competition - Intermediate Solution

This repository contains the `intermediate` ROS 2 package, which implements a Drone Swarm Controller and Exploration system for the **UAV competition at ICUAS 2026**. 

The package is designed to control a swarm of UAVs (Crazyflies) for autonomous exploration, mapping, and frontier detection in a simulated or physical environment.

## Overview

The `intermediate` package is built using ROS 2 (Python) and integrates directly with `crazyflie_interfaces`. It provides a suite of nodes for robust multi-agent control, including taking off, navigating to waypoints, managing trajectories, and cooperatively mapping unknown areas through frontier-based exploration.

### Key Features
- **Swarm Control:** Coordinated control of multiple Crazyflie drones.
- **Autonomous Exploration:** Frontier detection algorithms that allow the swarm to intelligently explore unknown areas.
- **Trajectory Control:** Smooth trajectory generation and execution using `drone_trajectory_controller`.
- **Mapping:** Octomap integration and 3D occupancy grid management for real-time environment representation.
- **Simulation:** Includes a custom OpenCV-based simulator (`icuas_cv2_simulator_v4`) for rapid testing without Gazebo.

## Package Structure and Nodes

The following executable nodes are exported by this package (as defined in `setup.py`):

- `takeoff_all`: Commands all drones in the swarm to take off simultaneously.
- `swarm_controller`: The central node orchestrating the multi-agent swarm behavior and task allocation.
- `frontier_detection`: Analyzes the map to find frontiers (boundaries between known and unknown space) to guide exploration.
- `exploration_map`: Maintains and updates the exploration map data structure.
- `drone_trajectory_controller`: Low-level trajectory control for individual drones.
- `map_publisher`: Publishes the environment map (e.g., from an Octomap).
- `transform`: Handles ArUco marker transformations and localization updates.
- `go`: Simple utility node to send a drone to a specific coordinate.
- `solution`: A minimal node demonstrating basic takeoff and waypoint navigation (`intermediate_solution.py`).

## Dependencies

- ROS 2 (Targeted for the ICUAS 2026 competition distribution)
- `rclpy`
- `crazyflie_interfaces`
- `octomap_msgs`
- `geometry_msgs`
- Standard Python libraries: `numpy`, `opencv-python` (for the CV2 simulator), etc.

## Setup and Building

Assuming you have a standard ROS 2 workspace (e.g., `~/ros2_ws/`):

```bash
# Clone this repository into your workspace src directory
cd ~/ros2_ws/src
# (Clone command here)

# Install any missing dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build the package
colcon build --packages-select intermediate

# Source the workspace
source install/setup.bash
```

## Running the Code

You can run individual nodes using the standard `ros2 run` command. For example:

```bash
# Take off all drones
ros2 run intermediate takeoff_all

# Start the swarm controller
ros2 run intermediate swarm_controller

# Run the frontier detection node
ros2 run intermediate frontier_detection
```

*(Note: Depending on the competition framework, these nodes are typically launched via a unified launch file provided by the competition organizers or added to this package).*
