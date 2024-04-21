# Autonomous Drone Navigation System

## Objective
The project aims to develop an autonomous drone navigation system capable of maneuvering through a simulated environment with obstacles. The primary focus is on achieving stable flight and efficient path planning in a complex arena.

## Key Components and Methodology

### Arena Creation
A simulated environment is designed with buildings serving as obstacles, creating a realistic setting for drone navigation.

### Drone Selection
The Crazyflie drone is utilized for its agility and suitability in a simulated environment.

### Stabilization
Implementation of a PID (Proportional, Integral, Derivative) controller to ensure the drone's stable flight amidst environmental variables.

### Point-to-Point Navigation
The drone is programmed to fly autonomously from one specified point to another within the arena.

### Node Class Development
A specialized Node class is created for managing the drone's navigation. This class is responsible for storing neighboring nodes, calculating GPS coordinates, and identifying obstacle nodes.

### Path Planning
The A* algorithm is employed for efficient path planning. This algorithm helps in determining the optimal route from the start point to the destination while avoiding obstacles.

### Path Following
The drone follows the path charted by the A* algorithm, moving from the start node to the goal node.

### Obstacle Sensing
The drone is equipped with sensors to detect obstacles in its path. Upon detection, it hovers, and the obstacle node is marked. Subsequently, the A* algorithm is reengaged to re-plan the path.

### Re-planned Path Navigation
After path re-planning, the drone resumes its journey, following the new route to the goal.

### Landing Upon Goal Achievement
Once the drone reaches the goal node, it is programmed to execute a safe landing.

## Conclusion
This project showcases the integration of advanced control systems, path planning algorithms, and real-time obstacle detection and avoidance in drone technology. The successful implementation of these elements in a simulated environment paves the way for real-world applications in various fields such as surveillance, delivery services, and disaster management.
