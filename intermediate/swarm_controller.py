import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, PoseStamped, Twist
from custom_interfaces.msg import Target, Targets
import numpy as np
import os
import math
from scipy.ndimage import binary_dilation

class APF_Swarm_Controller(Node):
    def __init__(self):
        super().__init__('apf_swarm_controller')

        # --- Constants from Paper (Tuned for Simulation) ---
        self.D_S = 1.0   # Safety distance (meters) [cite: 97]
        self.D_C = 3.0   # Max comms distance (meters) [cite: 96]
        self.SIGMA = 0.5 # Marr Wavelet dispersion width [cite: 103]
        self.K_SP = 0.1  # Shallow parabola gain [cite: 126]
        self.K_GOAL = 1.0 # Goal attraction gain
        self.K_OBS = 2.0  # Obstacle repulsion gain
        self.MAX_VEL = 0.5 # Cap velocity [cite: 51]
        
        # --- Existing Setup ---
        self.NUM_DRONES = int(os.environ.get('NUM_ROBOTS', 5))
        ocgrid_path = os.path.expanduser('~/ros2_ws/src/intermediate/intermediate/oc_grid.npy')
        self.ocgrid3d = np.load(ocgrid_path)
        self.HEIGHT = 3
        self.res = 0.15
        _, _, self.HEIGHT_IDX = self.world2grid(0, 0, self.HEIGHT)
        self.ocgrid = self.ocgrid3d[:, :, self.HEIGHT_IDX]

        self.cf_poses = {
            f"cf_{i}": {"x": None, "y": None, "yaw": None}
            for i in range(1, self.NUM_DRONES + 1)
        }

        self.global_pose_subscriptions = []
        for i in range(1, self.NUM_DRONES + 1):
            topic = f"/cf_{i}/pose"
            self.global_pose_subscriptions.append(
                self.create_subscription(PoseStamped, topic, 
                lambda msg, drone_id=i: self.global_pose_callback(msg, drone_id), 10)
            )

        self.target_sub = self.create_subscription(Targets, 'targets', self.target_callback, 10)
        self.targets = [] # List of (grid_x, grid_y)

        self.cmd_publishers = {}
        for i in range(1, self.NUM_DRONES + 1):
            self.cmd_publishers[f"cf_{i}"] = self.create_publisher(Twist, f"/cf_{i}/cmd_vel", 10)

        # Control Loop Timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def grid2world(self, x, y, z=0):
        # Converts grid indices to meters
        return ((x - 66.5)*self.res, (y - 66.5)*self.res, (z + 0.5) * self.res)

    def world2grid(self, x, y, z=0):
        # Converts meters to grid indices
        return (int(x / self.res + 66.5), int(y/ self.res + 66.5), int(z / self.res - 0.5))
    
    def global_pose_callback(self, msg: PoseStamped, drone_id):
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        
        # Store positions in WORLD frame (meters) for Physics calculations
        # NOTE: Changed from original snippet to store World coords for easier APF math
        self.cf_poses[f"cf_{drone_id}"]["x"] = msg.pose.position.x
        self.cf_poses[f"cf_{drone_id}"]["y"] = msg.pose.position.y
        self.cf_poses[f"cf_{drone_id}"]["yaw"] = yaw

    def target_callback(self, msg):
        self.targets = []
        for target in msg.targets:
            # Store targets in WORLD frame
            self.targets.append((target.x, target.y))

    # --- APF Helper Functions ---

    def get_range_force(self, p1, p2):
        """Calculates phi_range (Marr Wavelet + Shallow Parabola) """
        diff = p2 - p1
        dist = np.linalg.norm(diff)
        if dist == 0: return np.zeros(2)
        
        direction = diff / dist
        force = np.zeros(2)

        # 1. Marr Wavelet (Repulsion) if d < d_s [cite: 99]
        if dist < self.D_S:
            # Gradient of Marr Wavelet
            # Phi = (1 - d^2/sigma^2) * exp(-d^2 / 2sigma^2)
            # F = -Gradient. Roughly pushes away significantly at close range.
            # Simplified repulsive scaling for stability:
            mag = -1.0 * (1.0/dist - 1.0/self.D_S) 
            force = mag * direction

        # 2. Shallow Parabola (Attraction) if d > d_c [cite: 126]
        elif dist > self.D_C:
            # Phi = k * d^2 -> Gradient = 2 * k * d
            mag = 2 * self.K_SP * (dist - self.D_C)
            force = mag * direction
            
        # 3. Sweet spot (d_s <= d <= d_c) -> Force is 0 
        
        return force

    def check_los(self, p1, p2):
        """Raytrace on occupancy grid to check Line of Sight"""
        # Convert world points to grid indices
        g1 = self.world2grid(p1[0], p1[1])
        g2 = self.world2grid(p2[0], p2[1])
        
        x0, y0 = g1[0], g1[1]
        x1, y1 = g2[0], g2[1]
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < self.ocgrid.shape[0] and 0 <= y < self.ocgrid.shape[1]:
                    if self.ocgrid[x, y] == 1:
                        return False # Occlusion detected
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < self.ocgrid.shape[0] and 0 <= y < self.ocgrid.shape[1]:
                    if self.ocgrid[x, y] == 1:
                        return False # Occlusion detected
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True

    def get_los_force(self, p1, p2):
        """Calculates phi_LOS perpendicular to occlusion """
        if self.check_los(p1, p2):
            return np.zeros(2)
        
        # If blocked, force is perpendicular to vector p1->p2
        # to escape occlusion zone [cite: 152]
        vec = p2 - p1
        # Perpendicular vector (-y, x)
        perp = np.array([-vec[1], vec[0]])
        perp = perp / np.linalg.norm(perp)
        
        # We need to know which way is "out". 
        # Heuristic: Move towards the side with more free space or simply consistent rotation
        # The paper suggests using occlusion lines. Simplification: Rotate 90 deg.
        return perp * 1.5 # Arbitrary magnitude s

    def get_obstacle_force(self, pos):
        """Standard local obstacle repulsion"""
        grid_pos = self.world2grid(pos[0], pos[1])
        force = np.zeros(2)
        window = 5 # Check 5x5 grid around robot
        
        gx, gy = grid_pos[0], grid_pos[1]
        
        for i in range(-window, window+1):
            for j in range(-window, window+1):
                nx, ny = gx + i, gy + j
                if 0 <= nx < self.ocgrid.shape[0] and 0 <= ny < self.ocgrid.shape[1]:
                    if self.ocgrid[nx, ny] == 1:
                        # Repulsive force
                        obs_world = np.array(self.grid2world(nx, ny)[:2])
                        diff = pos - obs_world
                        dist = np.linalg.norm(diff)
                        if dist < 1.0 and dist > 0:
                            force += (diff / dist) * (1.0/dist)
        return force * self.K_OBS

    def control_loop(self):
        if not self.targets:
            return

        # 1. Update Topology (Count connections) [cite: 175]
        drone_ids = list(self.cf_poses.keys())
        connections = {did: [] for did in drone_ids}
        positions = {}

        for did in drone_ids:
            if self.cf_poses[did]["x"] is None: continue
            positions[did] = np.array([self.cf_poses[did]["x"], self.cf_poses[did]["y"]])

        # Build connectivity graph based on Range AND LOS [cite: 180]
        for i in range(len(drone_ids)):
            id_a = drone_ids[i]
            if id_a not in positions: continue
            
            for j in range(i+1, len(drone_ids)):
                id_b = drone_ids[j]
                if id_b not in positions: continue
                
                dist = np.linalg.norm(positions[id_a] - positions[id_b])
                has_los = self.check_los(positions[id_a], positions[id_b])
                
                # Connection exists if within max range and LOS [cite: 31, 33]
                if dist <= self.D_C and has_los:
                    connections[id_a].append(id_b)
                    connections[id_b].append(id_a)

        # 2. Compute Forces for each drone
        for i, did in enumerate(drone_ids):
            if did not in positions: continue
            
            p_curr = positions[did]
            
            # A. Mission Forces (Goal + Obstacle)
            target_idx = i % len(self.targets) # Assign target
            p_goal = np.array(self.targets[target_idx])
            
            f_goal = (p_goal - p_curr)
            # Normalize goal
            if np.linalg.norm(f_goal) > 0:
                f_goal = (f_goal / np.linalg.norm(f_goal)) * self.K_GOAL
            
            f_obs = self.get_obstacle_force(p_curr)
            
            v_high_priority = f_goal + f_obs # [cite: 173]

            # B. Communication Forces
            f_comm = np.zeros(2)
            
            # C. Parallel Composition Switch Logic 
            num_links = len(connections[did])
            
            # "Switch": If redundant (>= 2 links), ignore comm constraints [cite: 178]
            if num_links >= 2:
                 final_vel = v_high_priority
            else:
                # Not redundant: Compose with comm vectors 
                # Calculate recovery forces for ALL other robots (simplified from 'best pair')
                for other_did in drone_ids:
                    if did == other_did: continue
                    p_other = positions[other_did]
                    
                    # Range Force
                    f_comm += self.get_range_force(p_curr, p_other)
                    
                    # LOS Force (only if LOS is threatened/lost)
                    f_comm += self.get_los_force(p_curr, p_other)
                
                final_vel = v_high_priority + f_comm

            # D. Cap Velocity [cite: 51]
            speed = np.linalg.norm(final_vel)
            if speed > self.MAX_VEL:
                final_vel = (final_vel / speed) * self.MAX_VEL

            # Publish
            twist = Twist()
            twist.linear.x = float(final_vel[0])
            twist.linear.y = float(final_vel[1])
            self.cmd_publishers[did].publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = APF_Swarm_Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()