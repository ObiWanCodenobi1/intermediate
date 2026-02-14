import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, PoseStamped, Twist, Point, Vector3
from std_msgs.msg import ColorRGBA
from custom_interfaces.msg import Target, Targets
import numpy as np
import os
import math

class APF_Swarm_Controller(Node):
    def __init__(self):
        super().__init__('apf_swarm_controller')

        # --- Constants from Paper ---
        self.D_S = 1.0   # Safety distance [cite: 97]
        self.D_C = 3.0   # Max comms distance [cite: 96]
        self.SIGMA = 0.5 
        self.K_SP = 0.1  
        self.K_GOAL = 1.0 
        self.K_OBS = 2.0  
        self.MAX_VEL = 0.5 
        
        # --- Setup ---
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
        self.targets = [] 

        self.cmd_publishers = {}
        for i in range(1, self.NUM_DRONES + 1):
            self.cmd_publishers[f"cf_{i}"] = self.create_publisher(Twist, f"/cf_{i}/cmd_vel", 10)

        # --- Visualization Publisher ---
        self.vis_pub = self.create_publisher(MarkerArray, '/swarm_viz', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

    def grid2world(self, x, y, z=0):
        return ((x * self.res) - 66.5, (y * self.res) - 66.5, (z * self.res) + 0.5)

    def world2grid(self, x, y, z=0):
        return (int((x + 66.5) / self.res), int((y + 66.5) / self.res), int((z - 0.5) / self.res))
    
    def global_pose_callback(self, msg: PoseStamped, drone_id):
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        self.cf_poses[f"cf_{drone_id}"]["x"] = msg.pose.position.x
        self.cf_poses[f"cf_{drone_id}"]["y"] = msg.pose.position.y
        self.cf_poses[f"cf_{drone_id}"]["yaw"] = yaw

    def target_callback(self, msg):
        self.targets = []
        for target in msg.targets:
            self.targets.append((target.x, target.y))

    # --- APF Logic (Same as before) ---
    def get_range_force(self, p1, p2):
        diff = p2 - p1
        dist = np.linalg.norm(diff)
        if dist == 0: return np.zeros(2)
        direction = diff / dist
        force = np.zeros(2)

        # Marr Wavelet (Repulsion) [cite: 100]
        if dist < self.D_S:
            mag = -1.0 * (1.0/dist - 1.0/self.D_S) 
            force = mag * direction
        # Shallow Parabola (Attraction) [cite: 126]
        elif dist > self.D_C:
            mag = 2 * self.K_SP * (dist - self.D_C)
            force = mag * direction
            
        return force

    def check_los(self, p1, p2):
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
                        return False 
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
                        return False 
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True

    def get_los_force(self, p1, p2):
        if self.check_los(p1, p2):
            return np.zeros(2)
        vec = p2 - p1
        perp = np.array([-vec[1], vec[0]]) # Perpendicular Force [cite: 154]
        perp = perp / np.linalg.norm(perp)
        return perp * 1.5 

    def get_obstacle_force(self, pos):
        grid_pos = self.world2grid(pos[0], pos[1])
        force = np.zeros(2)
        window = 5 
        gx, gy = grid_pos[0], grid_pos[1]
        for i in range(-window, window+1):
            for j in range(-window, window+1):
                nx, ny = gx + i, gy + j
                if 0 <= nx < self.ocgrid.shape[0] and 0 <= ny < self.ocgrid.shape[1]:
                    if self.ocgrid[nx, ny] == 1:
                        obs_world = np.array(self.grid2world(nx, ny)[:2])
                        diff = pos - obs_world
                        dist = np.linalg.norm(diff)
                        if dist < 1.0 and dist > 0:
                            force += (diff / dist) * (1.0/dist)
        return force * self.K_OBS

    def control_loop(self):
        if not self.targets:
            return

        drone_ids = list(self.cf_poses.keys())
        connections = {did: [] for did in drone_ids}
        positions = {}
        
        # Visualization Data Containers
        viz_links = [] # List of (p1, p2) tuples
        viz_forces = [] # List of (start_pos, vector) tuples

        # 1. Update Topology
        for did in drone_ids:
            if self.cf_poses[did]["x"] is None: continue
            positions[did] = np.array([self.cf_poses[did]["x"], self.cf_poses[did]["y"]])

        for i in range(len(drone_ids)):
            id_a = drone_ids[i]
            if id_a not in positions: continue
            for j in range(i+1, len(drone_ids)):
                id_b = drone_ids[j]
                if id_b not in positions: continue
                
                dist = np.linalg.norm(positions[id_a] - positions[id_b])
                has_los = self.check_los(positions[id_a], positions[id_b])
                
                # Check for valid connection [cite: 31, 33]
                if dist <= self.D_C and has_los:
                    connections[id_a].append(id_b)
                    connections[id_b].append(id_a)
                    # Add to viz
                    viz_links.append((positions[id_a], positions[id_b]))

        # 2. Compute Forces
        for i, did in enumerate(drone_ids):
            if did not in positions: continue
            p_curr = positions[did]
            
            target_idx = i % len(self.targets)
            p_goal = np.array(self.targets[target_idx])
            
            f_goal = (p_goal - p_curr)
            if np.linalg.norm(f_goal) > 0:
                f_goal = (f_goal / np.linalg.norm(f_goal)) * self.K_GOAL
            
            f_obs = self.get_obstacle_force(p_curr)
            v_high_priority = f_goal + f_obs 

            f_comm = np.zeros(2)
            num_links = len(connections[did])
            
            # Switch Logic [cite: 175, 178]
            if num_links >= 2:
                 final_vel = v_high_priority
            else:
                for other_did in drone_ids:
                    if did == other_did: continue
                    p_other = positions[other_did]
                    f_comm += self.get_range_force(p_curr, p_other)
                    f_comm += self.get_los_force(p_curr, p_other)
                final_vel = v_high_priority + f_comm

            # Cap Velocity [cite: 51]
            speed = np.linalg.norm(final_vel)
            if speed > self.MAX_VEL:
                final_vel = (final_vel / speed) * self.MAX_VEL

            # Store for viz
            viz_forces.append((p_curr, final_vel))

            # Publish Control
            twist = Twist()
            twist.linear.x = float(final_vel[0])
            twist.linear.y = float(final_vel[1])
            self.cmd_publishers[did].publish(twist)

        # 3. Trigger Visualization
        self.publish_markers(positions, viz_links, viz_forces)

    def publish_markers(self, positions, links, forces):
        ma = MarkerArray()
        id_counter = 0

        # A. Draw Robots (Spheres)
        for did, pos in positions.items():
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "drones"
            m.id = id_counter
            id_counter += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = pos[0]
            m.pose.position.y = pos[1]
            m.pose.position.z = 0.5
            m.scale = Vector3(x=0.3, y=0.3, z=0.3)
            m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0) # Cyan
            ma.markers.append(m)

        # B. Draw Comm Links (Lines)
        # We use one LINE_LIST marker for all links for efficiency
        m_links = Marker()
        m_links.header.frame_id = "map"
        m_links.header.stamp = self.get_clock().now().to_msg()
        m_links.ns = "comm_links"
        m_links.id = 9999
        m_links.type = Marker.LINE_LIST
        m_links.action = Marker.ADD
        m_links.scale.x = 0.05 # Line width
        m_links.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8) # Green
        
        for p1, p2 in links:
            pt1 = Point(x=p1[0], y=p1[1], z=0.5)
            pt2 = Point(x=p2[0], y=p2[1], z=0.5)
            m_links.points.append(pt1)
            m_links.points.append(pt2)
        ma.markers.append(m_links)

        # C. Draw Forces (Arrows)
        for i, (pos, vec) in enumerate(forces):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "forces"
            m.id = id_counter
            id_counter += 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            
            # Start point
            p_start = Point(x=pos[0], y=pos[1], z=0.5)
            # End point (scaled by vector)
            p_end = Point(x=pos[0]+vec[0], y=pos[1]+vec[1], z=0.5)
            
            m.points = [p_start, p_end]
            m.scale.x = 0.05 # Shaft diameter
            m.scale.y = 0.1 # Head diameter
            m.scale.z = 0.1 # Head length
            m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0) # Yellow
            ma.markers.append(m)

        self.vis_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    controller = APF_Swarm_Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()