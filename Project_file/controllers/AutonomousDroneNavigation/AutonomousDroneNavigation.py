
import math
import heapq
import itertools  # For tie-breaking
from controller import Robot
from math import cos, sin
from pid_controller import VelocityHeightPIDController
from wall_following import WallFollowing


class Node:
    def __init__(self, row, col):
        """
        Initialize a new Node with specified row and column coordinates.
        """
        self.row = row
        self.col = col

    def __str__(self):
        """
        String representation of the Node.
        """
        return f"Node({self.row}, {self.col})"

    def is_diagonal_neighbor(self, other_node):
        """
        Check if another node is a diagonal neighbor.
        Diagonal neighbors are those where both the row and column differ by exactly 1.
        """
        dx = abs(self.row - other_node.row)
        dy = abs(self.col - other_node.col)
        return dx == 1 and dy == 1

    def calculate_distance(self, other_node):
        """
        Calculate the distance to another node.
        If the nodes are diagonal neighbors, use Euclidean distance.
        Otherwise, use a distance of 1 for horizontal/vertical neighbors and sqrt(2) for other diagonals.
        """
        if self.is_diagonal_neighbor(other_node):
            dx = abs(self.row - other_node.row)
            dy = abs(self.col - other_node.col)
            return math.sqrt(dx**2 + dy**2)
        else:
            return (
                1
                if (self.row == other_node.row or self.col == other_node.col)
                else math.sqrt(2)
            )

    def get_gps_coordinates(self, meter_scale, n):
        """
        Convert the node's grid position to GPS-like coordinates.
        The center of the grid is treated as (0,0), and each unit distance in the grid is scaled by 'meter_scale'.
        The grid size 'n' determines the offset for calculating coordinates.
        """
        if n % 2 != 0:
            even = 0
            x = (self.col - int(n / 2)) * meter_scale - even
            y = (-self.row + int(n / 2)) * meter_scale - even
            return (x, y)
        else:
            even = 0.5
            x = (self.col - int(n / 2)) * meter_scale - even
            y = (self.row - int(n / 2)) * meter_scale - even
            return (x, y)



# Initialize a 2D array of Node objects




def add_obstacle(node_map, obstacle_coordinates, obstacle_nodes):
    for obstacle_coordinate in obstacle_coordinates:
        x, y = obstacle_coordinate
        # Find the corresponding grid cell using search_node
        [row, col] = search_node(x, y)
        # Add the grid cell to the obstacle_nodes list
        if (row, col) not in obstacle_nodes:
            obstacle_nodes.append((row, col))
        # Mark the node as an obstacle in the node_map (optional)
        node_map[row][col].is_obstacle = True


def astar_path_planning(node_map, start_node, goal_node):
    """
    A* pathfinding algorithm to find the shortest path between two nodes in a grid.

    Parameters:
    node_map: A 2D grid representing the map.
    start_node: The starting node for the path.
    goal_node: The goal node for the path.
    """

    def heuristic(node1, node2):
        """
        Heuristic function to estimate the cost from the current node to the goal.
        Uses Euclidean distance as the heuristic.
        """
        x1, y1 = node1.get_gps_coordinates(1.0, len(node_map))
        x2, y2 = node2.get_gps_coordinates(1.0, len(node_map))
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx**2 + dy**2)

    def reconstruct_path(came_from, current_node):
        """
        Reconstructs the path from the start node to the current node.
        """
        path = [current_node]
        while current_node in came_from:
            current_node = came_from[current_node]
            path.insert(0, current_node)
        return path

    open_set = []  # Priority queue for nodes to be evaluated
    counter = itertools.count()  # Tie-breaking counter for nodes with equal f-scores

    # Add the start node to the open set with a cost of 0
    heapq.heappush(open_set, (0, next(counter), start_node))

    # Initialize g_score for each node: distance from start to the node
    g_score = {node: float("inf") for row in node_map for node in row}
    g_score[start_node] = 0

    came_from = {}  # For each node, which node it can most efficiently be reached from

    while open_set:
        # Pop the node with the lowest f-score from the open set
        _, _, current_node = heapq.heappop(open_set)

        # If the goal node is reached, reconstruct and return the path
        if current_node == goal_node:
            return [
                node.get_gps_coordinates(1.0, len(node_map))
                for node in reconstruct_path(came_from, goal_node)
            ]

        # For each neighbor of the current node
        for neighbor in get_neighbors(current_node, node_map):
            # Calculate tentative g_score for the neighbor
            tentative_g_score = g_score[current_node] + current_node.calculate_distance(neighbor)

            # If the tentative g_score is less than the g_score for the neighbor
            if tentative_g_score < g_score[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score, next(counter), neighbor))

    return None  # If no path is found, return None



# Rest of the code remains the same


def get_neighbors(current_node, node_map):
    """
    Get all valid neighbors of a given node in a grid.

    Parameters:
    current_node: The node for which neighbors are to be found.
    node_map: A 2D grid representing the map.

    Returns:
    A list of neighbor nodes.
    """

    neighbors = []  # List to store valid neighbors
    n = len(node_map)  # Size of the grid

    # Loop through a 5x5 grid centered on the current node
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            # Skip the current node itself
            if dx == 0 and dy == 0:
                continue

            # Calculate the row and column of the potential neighbor
            new_row, new_col = current_node.row + dx, current_node.col + dy

            # Check if the new row and column are within the grid boundaries
            if 0 <= new_row < n and 0 <= new_col < n:
                neighbor = node_map[new_row][new_col]

                # Check if the neighbor is not an obstacle
                if not is_obstacle(neighbor):
                    # If it's a valid neighbor, add it to the list
                    neighbors.append(neighbor)

    return neighbors

def adjust_for_boundary_wrap(angle_diff):
    """
    Adjust the angle difference for the 0°/360° boundary wrap-around.
    This function ensures that the angle difference is represented in the range [-180°, 180°].
    """
    if angle_diff > 180:
        return angle_diff - 360
    elif angle_diff < -180:
        return angle_diff + 360
    return angle_diff

def compute_yaw_change(yaw_m, yd2, yaw_change):
    """
    Compute the change in yaw (rotation) needed, considering the boundary wrap-around.
    """
    raw_diff = yd2 - yaw_m
    adjusted_diff = adjust_for_boundary_wrap(raw_diff)
    return yaw_change if adjusted_diff > 0 else -yaw_change



def fly(maintain_altitude, yd2, fl, stop, altitude, x_global, y_global, x, y, height_desired, sideways_desired, yaw_desired, forward_desired):
    """
    Function to control the flight of a drone.

    Parameters:
    - maintain_altitude: The altitude that the drone should maintain.
    - yd2: The desired yaw.
    - fl: Flag to indicate a certain state in the flight process.
    - stop: Flag to indicate whether the drone should stop.
    - altitude: The current altitude of the drone.
    - x_global, y_global: The global coordinates of the drone.
    - x, y: The local coordinates of the drone.
    - height_desired, sideways_desired, yaw_desired, forward_desired: Control parameters for the drone's movement.
    """
    sideways_desired = 0
    ee = math.sqrt((x_global - x) ** 2 + (y_global - y) ** 2)

    # Convert yaw to a value between 0 and 360 degrees
    if yaw >= 0:
        yaw_m = yaw * 60
    elif yaw < 0:
        yaw_m = 180 + (180 + yaw * 60)
    if yaw_m < 2 and yaw > 355:
        yaw_m = 0

    # Control logic for altitude, yaw, and position
    if altitude < maintain_altitude and fl == 0:  # First, reach the desired altitude
        height_desired += 0.5 * dt
    elif (abs(abs(yd2) - abs(yaw_m)) > 2) and fl == 0:  # Then, rotate to the desired angle
        yaw_change = 0.3
        yaw_desired += compute_yaw_change(yaw_m, yd2, yaw_change)
        forward_desired = 0
    elif altitude > maintain_altitude and fl == 0:
        yaw_desired = 0
        if ee > 0.2:  # Move to the specified x,y coordinate
            forward_desired += 0.2
        elif ee < 0.2:
            fl = 1
    elif ee < 0.5 and fl == 1:
        stop = 1
        fl = 0
    elif abs(abs(yd2) - abs(yaw_m)) > 5 or ee > 0.5:
        fl = 0

    return height_desired, sideways_desired, yaw_desired, forward_desired, fl, stop

def land(maintain_altitude, yd2, fl, stop, altitude, x_global, y_global, x, y, height_desired, sideways_desired, yaw_desired, forward_desired):
    """
    Function to control the landing of a vehicle (e.g., drone).

    Parameters:
    maintain_altitude: The altitude to maintain during the landing process.
    yd2, fl, stop: Control parameters (their specific roles are not clear from the context).
    altitude: Current altitude of the vehicle.
    x_global, y_global: Global position coordinates of the vehicle.
    x, y: Local position coordinates of the vehicle.
    height_desired: Desired height to achieve.
    sideways_desired: Desired sideways movement.
    yaw_desired: Desired yaw angle.
    forward_desired: Desired forward movement.
    """

    # Reset sideways movement to zero as it's a landing procedure
    sideways_desired = 0

    # Calculate yaw movement
    if yaw >= 0:
        yaw_m = yaw * 60
    elif yaw < 0:
        yaw_m = 180 + (180 + yaw * 60)

    # Normalize yaw movement
    if yaw_m < 2 and yaw > 355:
        yaw_m = 0

    # Adjust height to reach the desired maintain_altitude
    if altitude < maintain_altitude and fl == 0:  # Ascend if below maintain_altitude
        height_desired += 0.1 * dt
    if altitude > maintain_altitude and fl == 0:  # Descend if above maintain_altitude
        height_desired -= 0.1 * dt

    # Set yaw and forward movement to zero as it's a landing procedure
    yaw_desired = 0
    forward_desired = 0

    return height_desired, sideways_desired, yaw_desired, forward_desired, fl, stop

def search_node(x_required_to_start, y_required_to_start):
    """
    Function to find the closest node in a grid to a given set of coordinates.

    Parameters:
    x_required_to_start, y_required_to_start: The coordinates for which the closest node is to be found.
    """

    i = 0.0
    j = 0.0
    while True:
        # Get the GPS coordinates of the current node
        loc = node_map[int(i)][int(j)].get_gps_coordinates(1, n)

        # Check if the current node's x-coordinate is close enough to the required x-coordinate
        if not (abs(loc[0] - (x_required_to_start)) < 0.71):
            j += 1.0
            continue

        # Check if the current node's y-coordinate is close enough to the required y-coordinate
        if not (abs(loc[1] - (y_required_to_start)) < 0.71):
            i += 1.0
            continue

        # If both coordinates are close enough, break the loop
        else:
            break

    return [int(i), int(j)]

    
# n = 5
# rows, cols = n, n
# node = [[Node(i, j) for j in range(cols)] for i in range(rows)]


FLYING_ATTITUDE = 0
fl=0
# Main entry point of the script
if __name__ == '__main__':

    # Create an instance of the Robot class
    robot = Robot()
    # Set the basic time step of the world's simulation
    timestep = int(robot.getBasicTimeStep())

    # Initialize and configure motors
    # Motor 1 setup
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))  # Set position to infinity for velocity control
    m1_motor.setVelocity(-1)            # Set initial velocity

    # Motor 2 setup
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)

    # Motor 3 setup
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)

    # Motor 4 setup
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    # Initialize and enable sensors
    # Inertial Unit
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)

    # GPS
    gps = robot.getDevice("gps")
    gps.enable(timestep)

    # Gyroscope
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    # Camera
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Range finders for obstacle detection
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    # Initialize control and state variables
    # Variables for tracking position and time
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True
    stop = 0

    # Initialize PID controller for velocity control
    PID_crazyflie = VelocityHeightPIDController()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    # Set desired height and yaw for flight control
    height_desired = FLYING_ATTITUDE
    yaw_desired = 0
    

    
    # Initialize wall following behavior with specified parameters
    wall_following = WallFollowing(angle_value_buffer=0.01, reference_distance_from_wall=0.5,
                                   max_forward_speed=0.3, init_state=WallFollowing.StateWallFollowing.FORWARD)



    # Set the mode of operation (autonomous or manual)
    autonomous_mode = True  # Change to False for manual control

    # Utility function for range conversion
    def num_to_range(num, inMin, inMax, outMin, outMax):
        # Convert a number from one range to another
        return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

    # Define the grid size for path planning
    n = 101
    rows, cols = n, n
    
    # Create a 2D grid of nodes for path planning
    node_map = [[Node(i, j) for j in range(cols)] for i in range(rows)]
    
    # List to store nodes that are obstacles
    obstacle_nodes = []
    
    # Hardcoded obstacle coordinates
    obstacle_coordinates = []#(20.755617763765958, -20.530367055839346), (20.0, -20.0), (20.310253990881947, -20.54893299676575), (18.0, -18.0), (20.101197626780323, -20.5909426585509), (19.0, -18.0), (20.08065910975286, -20.528085134744405), (18.0, -19.0), (19.804392395333274, -20.607163001012946), (20.0, -18.0), (19.80843750433482, -20.552655553159475), (18.0, -20.0), (19.808436697407707, -20.55265480519727), (19.0, -19.0), (19.80843669740694, -20.552654805197378), (20.0, -19.0), (19.808436697406506, -20.55265480519742), (19.0, -20.0), (19.80843669740627, -20.55265480519743), (21.0, -18.0), (19.808436697406073, -20.552654805197434), (18.0, -21.0), (19.808436697405767, -20.55265480519744), (21.0, -19.0), (18.749027866270808, -20.53427672166952), (17.0, -19.0), (18.696527407558843, -20.48235173083074), (16.0, -18.0), (18.696527407492184, -20.4823517308775), (17.0, -18.0), (18.69652740743543, -20.482351730917177), (16.0, -19.0), (18.696527407386633, -20.482351730950153), (16.0, -20.0), (18.69652740730807, -20.482351730999504), (17.0, -20.0), (18.69652740726315, -20.4823517310264), (16.0, -21.0), (18.69652740723879, -20.48235173104111), (17.0, -21.0), (18.696527407218262, -20.482351731053896), (16.0, -22.0), (18.69652740718636, -20.482351731074633), (17.0, -22.0), (13.673525481491431, -20.503026907192396), (12.0, -19.0), (13.62199073554577, -20.450237851191005), (11.0, -18.0), (13.621977259302858, -20.450232917004197), (12.0, -18.0), (13.621956248389765, -20.450224571534257), (11.0, -19.0), (13.621944618718585, -20.450219441898472), (13.0, -18.0), (13.621938349952355, -20.450216583248686), (11.0, -20.0), (13.621928451308627, -20.450212222251754), (13.0, -19.0), (13.621922796288862, -20.45020992663212), (12.0, -20.0), (13.621919702132269, -20.450208708610386), (14.0, -18.0), (1.236252118829166, -11.112831978548655), (1.0, -11.0), (1.1754837261556892, -11.07822865525495), (-1.0, -9.0), (1.262865295674798, -11.229044155089644), (0.0, -9.0), (1.2335715198911985, -11.171603533652137), (1.0, -9.0), (1.2300518755464542, -11.167275870424298), (-1.0, -10.0), (1.2297375416681813, -11.167045227242314), (-1.0, -11.0), (1.229737541659261, -11.167045227245334), (0.0, -10.0), (1.229737541651676, -11.167045227247906), (1.0, -10.0), (1.2297375416451921, -11.167045227249975), (0.0, -11.0), (1.229737541634868, -11.167045227252821), (2.0, -9.0), (1.2297375416290004, -11.167045227254263), (3.0, -9.0), (1.2297375416258134, -11.167045227255057), (-1.0, -12.0), (1.2297375416231149, -11.167045227255775), (2.0, -10.0), (1.2159808287945024, -11.178973407075423), (0.0, -12.0), (1.2209161075803914, -8.12719870406602), (1.0, -8.0), (1.2234174439155758, -8.272912440002523), (-1.0, -6.0), (1.2655608732669248, -8.364270227404104), (0.0, -6.0), (1.2430229144350893, -8.297681222758873), (-1.0, -7.0), (1.2430231690070177, -8.297680619763659), (1.0, -6.0), (1.2430232998698842, -8.29768027528511), (-1.0, -8.0), (1.2430233656464102, -8.297680085960032), (0.0, -7.0), (1.2430234202680301, -8.297679924419313), (1.0, -7.0), (1.243023508622392, -8.297679670942854), (0.0, -8.0), (1.2430235624045916, -8.29767952943321), (2.0, -6.0), (1.1632733667688688, -4.0840016086853925), (-1.0, -2.0), (-12.641336247849624, 10.850799503446073), (-14.0, 12.0), (-12.687276997585215, 10.908569923953475), (-15.0, 13.0), (-12.687276997456985, 10.908569924126585), (-14.0, 13.0), (-12.687276997263261, 10.908569924404175), (-15.0, 12.0), (-12.687276997160566, 10.908569924564985), (-13.0, 13.0), (-12.687276997105943, 10.90856992465326), (-15.0, 11.0), (-12.687276997018175, 10.90856992479105), (-13.0, 12.0), (-12.68727699696627, 10.908569924867146), (-14.0, 11.0), (-12.687276996937548, 10.908569924908166), (-12.0, 13.0), (-12.687276996913091, 10.908569924943015), (-15.0, 10.0), (-12.687276996875292, 10.908569924998307), (-12.0, 12.0), (-12.687276996854473, 10.908569925029823), (-11.0, 13.0), (-12.687276996843263, 10.908569925047038), (-14.0, 10.0), (-12.687276996769032, 10.908569925135673), (-15.0, 9.0), (-12.634396105176922, 13.757409691652725), (-13.0, 14.0), (-12.626058902518114, 13.606620566254238), (-15.0, 16.0), (-12.63473807833816, 13.456752240892817), (-15.0, 15.0), (-12.609764784920133, 13.306981594429644), (-14.0, 15.0), (-12.64791897645522, 13.359533613069827), (-14.0, 16.0), (-12.596497901906938, 13.147078968894812), (-15.0, 14.0), (-12.563972427997498, 13.201490075355965), (-13.0, 15.0), (-12.569745498033207, 13.27385235999086), (-14.0, 14.0), (-12.632086576178763, 15.677832992816773), (-14.0, 17.0), (-12.67450257419137, 15.737468590239459), (-15.0, 18.0), (-12.67450257384123, 15.737468590740592), (-15.0, 17.0), (-12.674502573312664, 15.737468591544774), (-14.0, 18.0), (-12.674502573033118, 15.737468592010444), (-13.0, 18.0), (-12.674502572884649, 15.737468592265854), (-13.0, 17.0), (-12.674502572757195, 15.737468592482202), (-12.0, 18.0), (-12.674502572549526, 15.737468592817908), (-12.0, 17.0), (-12.674502572427714, 15.737468593004133), (-11.0, 18.0), (22.111554525254054, -20.607937965531978), (22.0, -20.0), (22.101625106712177, -20.535327450415082), (22.0, -18.0), (22.101624699153117, -20.53532687993664), (22.0, -19.0), (22.101624084850034, -20.53532596258959), (23.0, -18.0), (22.101623758683726, -20.5353254308162), (23.0, -19.0), (22.101623584694256, -20.535325139278545), (24.0, -18.0), (22.10162276916551, -20.535324003020722), (20.0, -22.0), (22.014742641382572, -20.549511617655895), (24.0, -19.0), (14.131995613284074, -20.609683428345093), (14.0, -19.0), (14.127228331959374, -20.536109969547873), (15.0, -18.0), (14.127216453654377, -20.536093251442846), (12.0, -21.0), (14.12721098494434, -20.53608783217793), (15.0, -19.0), (14.127210984943837, -20.536087832178065), (13.0, -21.0)]
    
    # Add obstacles to the node map
    add_obstacle(node_map, obstacle_coordinates, obstacle_nodes)
    
    # Function to check if a node is an obstacle
    def is_obstacle(node):
        return (node.row, node.col) in obstacle_nodes
    
    # Goal and start coordinates
    goal_x = -5
    goal_y = 42
    start_x = 30
    start_y = -10
    maintain_altitude = 1
    
    # Find the start node
    [i, j] = search_node(start_x, start_y)
    print(f"row ={i} col = {j}")
    start_node = node_map[i][j]
    
    # Find the goal node
    [i, j] = search_node(goal_x, goal_y)
    print(f"row ={i} col = {j}")
    goal_node = node_map[i][j]
    
    # Perform A* path planning
    path = astar_path_planning(node_map, start_node, goal_node)
    
    # Check if a path is found
    if path:
        print("Planned Path (GPS Coordinates):")
        # Uncomment below to print each node in the path
        # for node in path:
        #     print(node)
    else:
        # Default path if no path is found
        print("No path found.")
    
    # Initialize arrays to store x and y coordinates of the path
    dd = []
    ff = []
    
    # Extract coordinates from the path if it exists
    if path is not None:
        for i in range(len(path)):
            dd.append(path[i][0])  # Extract and store the x-coordinate
            ff.append(path[i][1])  # Extract and store the y-coordinate
    else:
        stop = 0
    
    # Initialize variables for the first coordinates in the path
    o = 0
    x = dd[0]
    y = ff[0]
    # Flags to indicate the presence of obstacles and the goal
    flag_obs = 0
    flag_goal = 0
    
    # Main control loop of the robot
    while robot.step(timestep) != -1:
        
        # Calculate the time difference since the last loop iteration
        dt = robot.getTime() - past_time
        # Dictionary to store the current state of the robot
        actual_state = {}
    
        # Initialize the global position and time during the first iteration
        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False
    
        # Get sensor data
        # Roll, pitch, and yaw from the Inertial Measurement Unit (IMU)
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
    
        # Yaw rate from the gyroscope
        yaw_rate = gyro.getValues()[2]
    
        # Global position from GPS and calculate global velocities
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global) / dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global) / dt
        altitude = gps.getValues()[2]
        
        # Get range sensor values in meters
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000
        range_back_value = range_back.getValue() / 1000
    
        # Convert global velocities to body-fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw
                    
        # Initialize desired state variables
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0
            
        # Threshold distance for obstacle detection
        hit_point = 0.3
        
        # Check if any of the range sensors detect an obstacle within the hit point distance
        if (range_front_value < hit_point or range_right_value < hit_point or range_left_value < hit_point):
        
            # Check if this is the first time an obstacle is detected
            if flag_obs == 0:
                # Add the current GPS coordinates to the list of obstacle coordinates if not already present
                if (x_global, y_global) not in obstacle_coordinates:
                    obstacle_coordinates.append((x_global, y_global))
                    add_obstacle(node_map, [(x_global, y_global)], obstacle_nodes)
        
            # Add the current target coordinates to the list of obstacle coordinates if not already present
            if (x, y) not in obstacle_coordinates:
                obstacle_coordinates.append((x, y))
                add_obstacle(node_map, [(x, y)], obstacle_nodes)
        
            # Uncomment below to print each node in obstacle_nodes
            # for node in obstacle_nodes:
            #     print(node)
        
            # Set the wall following direction and get the side range value
            direction = WallFollowing.WallFollowingDirection.LEFT
            range_side_value = range_right_value
        
            # Get the velocity commands from the wall following state machine
            cmd_vel_x, cmd_vel_y, cmd_ang_w, state_wf = wall_following.wall_follower(
            range_front_value, range_side_value, yaw, direction, robot.getTime())

            # If in autonomous mode, set the desired velocities and angular velocity
            if autonomous_mode:
                sideways_desired = cmd_vel_y
                forward_desired = cmd_vel_x
                yaw_desired = cmd_ang_w
        
            # Use the PID controller to calculate motor power for maintaining desired velocities and height
            motor_power = PID_crazyflie.compute_pid(dt, forward_desired, sideways_desired,
                                            yaw_desired, height_desired,
                                            roll, pitch, yaw_rate,
                                            altitude, v_x, v_y)
        
            # Set motor velocities based on the calculated motor power
            m1_motor.setVelocity(-motor_power[0])
            m2_motor.setVelocity(motor_power[1])
            m3_motor.setVelocity(-motor_power[2])
            m4_motor.setVelocity(motor_power[3])
        
            # Update the past time and global position for the next iteration
            past_time = robot.getTime()
            past_x_global = x_global
            past_y_global = y_global
            # Set the flag to indicate an obstacle has been detected
            flag_obs = 1
            continue  # Continue to the next iteration of the loop

        
        # Check if an obstacle was previously detected
        if (flag_obs == 1):
            # Reset the obstacle detection flag
            flag_obs = 0
        
            # Find the node in the node map corresponding to the robot's current global position
            [i, j] = search_node(x_global, y_global)
            print(f"row ={i} col = {j}")
        
            # Set this node as the new start node for path planning
            start_node = node_map[i][j]
        
            # Find the node in the node map corresponding to the goal position
            [i, j] = search_node(goal_x, goal_y)
            print(f"row ={i} col = {j}")
        
            # Set this node as the goal node for path planning
            goal_node = node_map[i][j]
        
            # Perform A* path planning from the new start node to the goal node
            path = astar_path_planning(node_map, start_node, goal_node)
        
            # Initialize arrays to store x and y coordinates of the new path
            dd = []
            ff = []
        
            # If a path is found, extract and store the x and y coordinates
            if path is not None:
                for i in range(len(path)):
                    dd.append(path[i][0])  # Extract and store the x-coordinate
                    ff.append(path[i][1])  # Extract and store the y-coordinate
            else:
                # If no path is found, set the stop flag
                stop = 0
        
            # Set the first waypoint from the new path as the next target
            o = 1
            x = dd[1]
            y = ff[1]
            # Uncomment below to print the next target coordinates
            # print(f"{x} x y {y}")

        # Initialize desired state and movement variables
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0
        
        # Check if the robot has reached the last waypoint
        if o >= len(dd) - 1:
            stop = 0
            flag_goal = 1  # Set the flag indicating the goal has been reached
        
        # If the robot has not stopped, update the next waypoint
        if stop != 0:
            o = o + 1
            stop = 0
            x = dd[o]  # Update the x-coordinate of the next waypoint
            y = ff[o]  # Update the y-coordinate of the next waypoint
        
        # Update the desired height
        height_desired += height_diff_desired * dt
        
        # Calculate the angle to the next waypoint
        angle = math.degrees(math.atan2((y - y_global), (x - x_global)))
        yd2 = angle if angle > 0 else 180 + (180 + angle)
        
        # Convert the angle to a range suitable for the robot's control system
        if angle > 0:
            yd = num_to_range(angle, 0, 180, 0, 3)
        else:
            yd = num_to_range(angle, 0, -180, 0, -3)
        
        # Check if the goal has been reached to initiate landing
        if flag_goal == 1:
            print("landing")
            # Uncomment below to print the obstacle coordinates
            # print(obstacle_coordinates)
        
            # Gradually decrease the altitude for landing
            if maintain_altitude > 0:
                maintain_altitude -= 0.001
        
            # Set the current position as the target for landing
            x = x_global
            y = y_global
        
            # Call the landing function to get the desired states for landing
            (height_desired, sideways_desired, yaw_desired, forward_desired, fl, stop,) = land(
                maintain_altitude, yd2, fl, stop, altitude, x_global, y_global, x, y,
                height_desired, sideways_desired, yaw_desired, forward_desired)
        
            # Use the PID controller to calculate motor power for landing
            motor_power = PID_crazyflie.compute_pid(dt, forward_desired, sideways_desired,
                                            yaw_desired, height_desired,
                                            roll, pitch, yaw_rate,
                                            altitude, v_x, v_y)
        
            # Set motor velocities based on the calculated motor power
            m1_motor.setVelocity(-motor_power[0])
            m2_motor.setVelocity(motor_power[1])
            m3_motor.setVelocity(-motor_power[2])
            m4_motor.setVelocity(motor_power[3])
        
            # Update the past time and global position for the next iteration
            past_time = robot.getTime()
            past_x_global = x_global
            past_y_global = y_global
        
            continue  # Continue to the next iteration of the loop

        # Check if the robot didnt detected any obstacle or also didnt reached the goal
        if stop == 0:
        
            # Call the fly function to get the desired states for flying
            (height_desired, sideways_desired, yaw_desired, forward_desired, fl, stop,) = fly(
                maintain_altitude, yd2, fl, stop, altitude, x_global, y_global, x, y,
                height_desired, sideways_desired, yaw_desired, forward_desired)
        
        # Use the PID controller to calculate motor power for flying
        motor_power = PID_crazyflie.compute_pid(dt, forward_desired, sideways_desired,
                                                yaw_desired, height_desired,
                                                roll, pitch, yaw_rate,
                                                altitude, v_x, v_y)
        
        # Set motor velocities based on the calculated motor power
        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])
        
        # Update the past time and global position for the next iteration
        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
        
        