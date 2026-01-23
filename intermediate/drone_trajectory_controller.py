import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger
from crazyflie_interfaces.srv import UploadTrajectory, StartTrajectory  # Assuming Crazyflie services
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import os
import numpy as np
import math
from collections import deque

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_i] += 1
            return True
        return False

class DroneTrajectoryController(Node):
    def __init__(self):
        super().__init__('drone_trajectory_controller')

        self.NUM_DRONES = int(os.environ.get('NUM_ROBOTS', 10))
        self.MAX_COMM_DIST = 40
        self.SENSOR_RADIUS = 3
        self.ocgrid3d = np.load('oc_grid.npy')
        self.HEIGHT = 3
        _, _, self.HEIGHT_IDX = self.world2grid(0, 0, self.HEIGHT)
        self.ocgrid = self.ocgrid3d[:, :, self.HEIGHT_IDX]

        # Constants
        self.OBSTACLE_RADIUS = 4
        self.GRID_CELLS = self.ocgrid.shape[0]
        self.SIGMA = 3.0
        self.G_GAIN = 8.0
        self.MAX_FORCE_STEP = 3
        self.ZERO_RADIUS = 1
        self.EPS = 1e-3
        self.T_REDUCE = 0.90
        self.T_ZERO = 0.95
        self.ALPHA = 0.1
        self.MAX_RELOCATION_STEPS = 100
        self.RESTORATION_COOLDOWN_TICKS = 20
        self.RESTORATION_RECURSION_LIMIT = 5
        self.MAX_MOVE_PER_DRONE = 3
        self.PDR_TRIGGER_THRESHOLD = 0.9 * self.MAX_COMM_DIST

        # Phase Control
        self.phase = "DEPLOYMENT"
        self.convergence_counter = 0
        self.prev_score = 0
        self.restoration_active = False
        self.restoration_cooldown = 0
        self.relocation_plan = []

        # Exploration Maps
        self.visited_penalty = np.zeros(self.ocgrid.shape)
        self.exploration_reward = np.zeros(self.ocgrid.shape)
        self.exploration_map = np.zeros(self.ocgrid.shape)

        # Grid coord cache
        self.xx, self.yy = np.meshgrid(np.arange(self.ocgrid.shape[0]), np.arange(self.ocgrid.shape[1]))

        self.cf_poses = {
            f"cf_{i}": {
                "position": {"x": None, "y": None, "z": None},
                "orientation": {"x": None, "y": None, "z": None, "w": None}
            }
            for i in range(1, self.NUM_DRONES + 1)
        }

        self.agv_pose_sub = self.create_subscription(Point, '/AGV/pose', self.agv_pose_callback, 10)

        self.global_pose_subscriptions = []
        for i in range(1, self.NUM_DRONES + 1):
            self.global_pose_subscriptions.append(self.create_subscription(PoseStamped, f"/cf_{i}/pose", lambda msg, drone_id=i: self.global_pose_callback(msg, drone_id), 10))

        # Visualization publishers
        self.drone_marker_pub = self.create_publisher(MarkerArray, '/drone_markers', 10)
        self.trajectory_marker_pub = self.create_publisher(MarkerArray, '/trajectory_markers', 10)
        self.connectivity_marker_pub = self.create_publisher(MarkerArray, '/connectivity_markers', 10)
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        self.exploration_marker_pub = self.create_publisher(Marker, '/exploration_map', 10)

        # Service clients
        self.upload_clients = {}
        self.start_clients = {}
        self.callback_group = ReentrantCallbackGroup()
        for i in range(1, self.NUM_DRONES + 1):
            self.upload_clients[i] = self.create_client(UploadTrajectory, f'/cf_{i}/upload_trajectory', callback_group=self.callback_group)
            self.start_clients[i] = self.create_client(StartTrajectory, f'/cf_{i}/start_trajectory', callback_group=self.callback_group)

        self.p1x, self.p1y, _ = self.world2grid(-5, -5, 0)
        self.p1 = (self.p1x, self.p1y)
        self.p2 = None
        self.drone_positions = [(0, 0) for _ in range(self.NUM_DRONES)]

        # Setup environment
        self._setup_environment()

        # Timer for periodic updates
        self.timer = self.create_timer(0.1, self.step, callback_group=self.callback_group)

    def _setup_environment(self):
        print("Setting up environment...")
        occupied = np.where(self.ocgrid > 0.5)
        self.obstacles = list(zip(occupied[0], occupied[1]))
        self.NUM_OBSTACLES = len(self.obstacles)

        self.g_masks = np.ones((self.NUM_OBSTACLES, self.GRID_CELLS, self.GRID_CELLS), dtype=float)
        self.total_cells = np.zeros(self.NUM_OBSTACLES, dtype=float)
        self.visited_cells = np.zeros(self.NUM_OBSTACLES, dtype=float)

        self.obstacle_grid = self.ocgrid > 0.5

        gauss_list = []
        for idx, (obs_x, obs_y) in enumerate(self.obstacles):
            dx = self.xx - obs_x
            dy = self.yy - obs_y
            dist = np.hypot(dx, dy)
            pit = np.exp(-(dist ** 2) / (2 * self.SIGMA ** 2))

            ring_mask = (dist > self.OBSTACLE_RADIUS) & (dist <= self.OBSTACLE_RADIUS + 3)
            self.total_cells[idx] = float(np.count_nonzero(ring_mask))
            gauss_list.append(pit)

        self.gauss_stack = np.stack(gauss_list, axis=0)
        self.height_map = -np.sum(self.gauss_stack * self.g_masks, axis=0)
        self.los_cache = {}
        print("Environment setup complete.")
        self._publish_obstacle_markers()  # Publish obstacles once at setup

    def _publish_obstacle_markers(self):
        marker_array = MarkerArray()
        for i, (obs_x, obs_y) in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            wx, wy, wz = self.grid2world(obs_x, obs_y, 0)
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = wz + 1.0  # Half height above ground
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.res * 2  # Size of obstacle
            marker.scale.y = self.res * 2
            marker.scale.z = 2.0  # Height
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            marker_array.markers.append(marker)
        self.obstacle_marker_pub.publish(marker_array)

    def _publish_drone_markers(self):
        marker_array = MarkerArray()
        for i in range(self.NUM_DRONES):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'drones'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            pos = self.drone_positions[i]
            wx, wy, wz = self.grid2world(pos[0], pos[1], self.HEIGHT)
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = wz
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            # Color based on connectivity
            groups = self._get_connected_components(self.drone_positions)
            group_id = None
            for gid, members in groups.items():
                if i in members:
                    group_id = gid
                    break
            if group_id is not None:
                # Use different colors for different groups
                colors = [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
                color = colors[group_id % len(colors)]
                marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            marker_array.markers.append(marker)
        self.drone_marker_pub.publish(marker_array)

    def _publish_connectivity_markers(self):
        marker_array = MarkerArray()
        nodes = self.drone_positions + [self.p1, self.p2]
        marker_id = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._has_los(nodes[i], nodes[j]):
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'connectivity'
                    marker.id = marker_id
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.05  # Line width
                    marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5)
                    marker.points = []
                    # Start point
                    p1 = Point()
                    if i < self.NUM_DRONES:
                        wx1, wy1, wz1 = self.grid2world(nodes[i][0], nodes[i][1], self.HEIGHT)
                    else:
                        wx1, wy1, wz1 = self.grid2world(nodes[i][0], nodes[i][1], 0)
                    p1.x, p1.y, p1.z = wx1, wy1, wz1
                    marker.points.append(p1)
                    # End point
                    p2 = Point()
                    if j < self.NUM_DRONES:
                        wx2, wy2, wz2 = self.grid2world(nodes[j][0], nodes[j][1], self.HEIGHT)
                    else:
                        wx2, wy2, wz2 = self.grid2world(nodes[j][0], nodes[j][1], 0)
                    p2.x, p2.y, p2.z = wx2, wy2, wz2
                    marker.points.append(p2)
                    marker_array.markers.append(marker)
                    marker_id += 1
        self.connectivity_marker_pub.publish(marker_array)

    def _publish_trajectory_markers(self, trajectories):
        marker_array = MarkerArray()
        marker_id = 0
        for drone_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = 'trajectories'
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1  # Line width
                marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)
                marker.points = []
                for pose in trajectory:
                    p = Point()
                    p.x = pose.pose.position.x
                    p.y = pose.pose.position.y
                    p.z = pose.pose.position.z
                    marker.points.append(p)
                marker_array.markers.append(marker)
                marker_id += 1
        self.trajectory_marker_pub.publish(marker_array)

    def _publish_exploration_map(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'exploration'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.res
        marker.scale.y = self.res
        marker.color = ColorRGBA(r=0.0, g=0.5, b=0.0, a=0.3)
        marker.points = []
        for x in range(self.GRID_CELLS):
            for y in range(self.GRID_CELLS):
                if self.exploration_map[y, x] > 0.5:
                    p = Point()
                    wx, wy, wz = self.grid2world(x, y, 0)
                    p.x, p.y, p.z = wx, wy, wz
                    marker.points.append(p)
        self.exploration_marker_pub.publish(marker)

    def grid2world(self, x, y, z):
        return ((x * self.res) - 66.5, (y * self.res) - 66.5, (z * self.res) + 0.5)

    def world2grid(self, x, y, z):
        self.res = 0.15  # Assuming resolution
        return (int((x + 66.5) / self.res), int((y + 66.5) / self.res), int((z - 0.5) / self.res))

    def agv_pose_callback(self, msg):
        grid_x, grid_y, _ = self.world2grid(msg.x, msg.y, 0)
        self.p2 = (grid_x, grid_y)

    def global_pose_callback(self, msg: PoseStamped, drone_id):
        self.cf_poses[f"cf_{drone_id}"]["position"]["x"] = msg.pose.position.x
        self.cf_poses[f"cf_{drone_id}"]["position"]["y"] = msg.pose.position.y
        self.cf_poses[f"cf_{drone_id}"]["position"]["z"] = msg.pose.position.z

        self.cf_poses[f"cf_{drone_id}"]["orientation"]["x"] = msg.pose.orientation.x
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["y"] = msg.pose.orientation.y
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["z"] = msg.pose.orientation.z
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["w"] = msg.pose.orientation.w

        grid_x, grid_y, _ = self.world2grid(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self.drone_positions[drone_id - 1] = (grid_x, grid_y)

    def _has_los(self, pos1, pos2):
        key = (min(pos1, pos2), max(pos1, pos2))
        if key in self.los_cache:
            return self.los_cache[key]

        if math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]) > self.MAX_COMM_DIST:
            self.los_cache[key] = False
            return False

        steps = 10
        for t in np.linspace(0, 1, steps):
            ix = int(pos1[0] + t * (pos2[0] - pos1[0]))
            iy = int(pos1[1] + t * (pos2[1] - pos1[1]))
            if 0 <= ix < self.GRID_CELLS and 0 <= iy < self.GRID_CELLS:
                if self.obstacle_grid[iy, ix]:
                    self.los_cache[key] = False
                    return False

        self.los_cache[key] = True
        return True

    def _is_valid(self, pos):
        if not (0 <= pos[0] < self.GRID_CELLS and 0 <= pos[1] < self.GRID_CELLS):
            return False
        return not self.obstacle_grid[pos[1], pos[0]]

    def _get_connected_components(self, positions):
        nodes = positions + [self.p1, self.p2]
        n = len(nodes)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if self._has_los(nodes[i], nodes[j]):
                    uf.union(i, j)
        groups = {}
        for i in range(n):
            root = uf.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups

    def _calculate_exploration_score(self, positions):
        groups = self._get_connected_components(positions)
        if len(groups) > 1:
            return -1e6 * len(groups)
        total_score = 0
        for pos in positions:
            total_score += -self.height_map[pos[1]][pos[0]]
            total_score -= self.visited_penalty[pos[1]][pos[0]] * 0.5
        total_score -= self._comm_soft_penalty(positions)
        return total_score

    def _comm_soft_penalty(self, positions):
        nodes = positions + [self.p1, self.p2]
        safety_thresh = 0.8 * self.MAX_COMM_DIST
        penalty = 0.0
        for i, n in enumerate(nodes):
            min_dist = float('inf')
            for j, m in enumerate(nodes):
                if i != j:
                    dist = math.hypot(n[0] - m[0], n[1] - m[1])
                    if dist < min_dist:
                        min_dist = dist
            if min_dist == float('inf'):
                penalty += 1000
            elif min_dist > safety_thresh:
                penalty += (min_dist - safety_thresh) ** 2
        return penalty

    def _apply_visit_effects(self, positions):
        for pos in positions:
            self.visited_penalty[pos[1]][pos[0]] += 0.2
            r = self.ZERO_RADIUS
            x_min = max(0, pos[0] - r)
            x_max = min(self.GRID_CELLS, pos[0] + r + 1)
            y_min = max(0, pos[1] - r)
            y_max = min(self.GRID_CELLS, pos[1] + r + 1)
            self.exploration_map[y_min:y_max, x_min:x_max] = 1

            for i in range(self.NUM_OBSTACLES):
                obs_x, obs_y = self.obstacles[i]
                dx = self.xx[y_min:y_max, x_min:x_max] - obs_x
                dy = self.yy[y_min:y_max, x_min:x_max] - obs_y
                dist = np.hypot(dx, dy)
                ring_mask = (dist > self.OBSTACLE_RADIUS) & (dist <= self.OBSTACLE_RADIUS + 3)
                self.visited_cells[i] += np.sum(ring_mask)
                self.g_masks[i, y_min:y_max, x_min:x_max] = np.where(ring_mask, 0.0, self.g_masks[i, y_min:y_max, x_min:x_max])

        self.height_map = -np.sum(self.gauss_stack * self.g_masks, axis=0)

    def _compute_per_obstacle_fraction(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            f = np.divide(self.visited_cells, self.total_cells, out=np.zeros_like(self.visited_cells), where=self.total_cells > 0)
        return np.clip(f, 0.0, 1.0)

    def _g_scale_for_fraction(self, f):
        if f >= self.T_ZERO:
            return 0.01
        if f >= self.T_REDUCE:
            return 0.10
        return 1.0

    def _compute_force_vector(self, pos):
        fx = 0.0
        fy = 0.0
        f_obs = self._compute_per_obstacle_fraction()
        for idx, (ox, oy) in enumerate(self.obstacles):
            mask_val = self.g_masks[idx, pos[1], pos[0]]
            if mask_val <= 0:
                continue
            scale = self._g_scale_for_fraction(f_obs[idx])
            if scale <= 0:
                continue
            dx = ox - pos[0]
            dy = oy - pos[1]
            dist = math.hypot(dx, dy) + self.EPS
            mag = self.G_GAIN * scale * (1.0 / dist)
            fx += (dx / dist) * mag * mask_val
            fy += (dy / dist) * mag * mask_val
        return fx, fy

    def _add_force_candidate(self, candidates, curr_pos, idx):
        fx, fy = self._compute_force_vector(curr_pos[idx])
        norm = math.hypot(fx, fy)
        if norm < 1e-6:
            return
        dir_x = fx / norm
        dir_y = fy / norm
        step_len = min(self.MAX_FORCE_STEP, max(1.0, norm))
        for _ in range(3):
            nx = int(round(curr_pos[idx][0] + dir_x * step_len))
            ny = int(round(curr_pos[idx][1] + dir_y * step_len))
            if self._is_valid((nx, ny)):
                candidates.append((curr_pos[:idx] + [(nx, ny)] + curr_pos[idx + 1:], -self.height_map[ny][nx]))
                break
            step_len = max(1.0, step_len * 0.5)

    def enter_restoration_mode(self):
        if not self.restoration_active:
            self.restoration_active = True
            self.get_logger().info("Entering RESTORATION mode")

    def exit_restoration_mode(self):
        if self.restoration_active:
            self.restoration_active = False
            self.restoration_cooldown = self.RESTORATION_COOLDOWN_TICKS
            self.relocation_plan = []
            self.get_logger().info("Resuming EXPLORATION")

    def _solve_cascade(self, idx, target_pos, current_positions, moved_indices, depth=0):
        if depth > self.RESTORATION_RECURSION_LIMIT:
            return []

        if idx in moved_indices:
            return []

        curr_pos = current_positions[idx]
        delta_x = target_pos[0] - curr_pos[0]
        delta_y = target_pos[1] - curr_pos[1]
        dist = math.hypot(delta_x, delta_y)

        step = min(self.MAX_MOVE_PER_DRONE, dist)
        if step < 1.0:
            return []

        nx = int(curr_pos[0] + (delta_x / dist) * step)
        ny = int(curr_pos[1] + (delta_y / dist) * step)

        if not self._is_valid((nx, ny)):
            return []

        proposed_moves = [(idx, (nx, ny))]
        temp_moved_indices = moved_indices | {idx}

        nodes = current_positions + [self.p1, self.p2]
        my_neighbors = []
        for i, pos_i in enumerate(nodes):
            if i == idx:
                continue
            if self._has_los(nodes[idx], pos_i):
                my_neighbors.append(i)

        for n_idx in my_neighbors:
            pos_n = nodes[n_idx]
            if not self._has_los((nx, ny), pos_n):
                if n_idx >= self.NUM_DRONES:
                    continue
                cascade = self._solve_cascade(n_idx, (nx, ny), current_positions, temp_moved_indices, depth + 1)
                if not cascade:
                    return []
                proposed_moves.extend(cascade)
                for m_idx, _ in cascade:
                    temp_moved_indices.add(m_idx)

        return proposed_moves

    def _plan_relocation_step(self):
        groups = self._get_connected_components(self.drone_positions)
        if len(groups) <= 1:
            return None

        group_ids = list(groups.keys())
        best_pair_val = float('inf')
        best_merge_move = None
        nodes = self.drone_positions + [self.p1, self.p2]

        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                g1 = groups[group_ids[i]]
                g2 = groups[group_ids[j]]

                for u in g1:
                    if u >= self.NUM_DRONES:
                        continue
                    for v in g2:
                        if v >= self.NUM_DRONES:
                            continue
                        dist = math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])
                        if dist < best_pair_val:
                            best_pair_val = dist
                            best_merge_move = (u, nodes[v])

        if best_merge_move:
            idx, target = best_merge_move
            plan = self._solve_cascade(idx, target, self.drone_positions, set())
            return plan
        return None

    async def upload_trajectory(self, drone_id, trajectory):
        client = self.upload_clients[drone_id]
        req = UploadTrajectory.Request()
        req.trajectory = trajectory  # Assuming trajectory is a list of PoseStamped
        future = client.call_async(req)
        await future
        if future.result() is not None:
            self.get_logger().info(f"Trajectory uploaded for cf_{drone_id}")
        else:
            self.get_logger().error(f"Failed to upload trajectory for cf_{drone_id}")

    async def start_trajectory(self, drone_id):
        client = self.start_clients[drone_id]
        req = StartTrajectory.Request()
        future = client.call_async(req)
        await future
        if future.result() is not None:
            self.get_logger().info(f"Trajectory started for cf_{drone_id}")
        else:
            self.get_logger().error(f"Failed to start trajectory for cf_{drone_id}")

    def step(self):
        if self.p2 is None:
            return

        groups = self._get_connected_components(self.drone_positions)

        if len(groups) > 1:
            self.enter_restoration_mode()
        else:
            if self.restoration_cooldown > 0:
                self.restoration_cooldown -= 1
            else:
                self.exit_restoration_mode()

        if self.restoration_active:
            plan = self._plan_relocation_step()
            if plan:
                for idx, (tx, ty) in plan:
                    self.drone_positions[idx] = (tx, ty)
                    # Create simple trajectory: current to target
                    trajectory = []
                    current = self.drone_positions[idx]
                    steps = 10  # Simple linear interpolation
                    for s in range(steps + 1):
                        t = s / steps
                        ix = int(current[0] + t * (tx - current[0]))
                        iy = int(current[1] + t * (ty - current[1]))
                        wx, wy, wz = self.grid2world(ix, iy, self.HEIGHT)
                        pose = PoseStamped()
                        pose.header.frame_id = 'map'
                        pose.pose.position.x = wx
                        pose.pose.position.y = wy
                        pose.pose.position.z = wz
                        trajectory.append(pose)
                    # Upload and start
                    import asyncio
                    asyncio.create_task(self.upload_trajectory(idx + 1, trajectory))
                    asyncio.create_task(self.start_trajectory(idx + 1))
        else:
            candidates = []
            for idx in range(self.NUM_DRONES):
                self._add_force_candidate(candidates, self.drone_positions, idx)

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_positions = candidates[0][0]
                for idx, pos in enumerate(best_positions):
                    if pos != self.drone_positions[idx]:
                        self.drone_positions[idx] = pos
                        # Create trajectory
                        trajectory = []
                        current = self.drone_positions[idx]
                        steps = 10
                        for s in range(steps + 1):
                            t = s / steps
                            ix = int(current[0] + t * (pos[0] - current[0]))
                            iy = int(current[1] + t * (pos[1] - current[1]))
                            wx, wy, wz = self.grid2world(ix, iy, self.HEIGHT)
                            pose = PoseStamped()
                            pose.header.frame_id = 'map'
                            pose.pose.position.x = wx
                            pose.pose.position.y = wy
                            pose.pose.position.z = wz
                            trajectory.append(pose)
                        # Upload and start
                        import asyncio
                        asyncio.create_task(self.upload_trajectory(idx + 1, trajectory))
                        asyncio.create_task(self.start_trajectory(idx + 1))

            self._apply_visit_effects(self.drone_positions)

        # Publish visualizations
        self._publish_obstacle_markers()
        self._publish_drone_markers()
        self._publish_connectivity_markers()
        self._publish_exploration_map()
        # Note: Trajectory markers would be published when trajectories are created, but for simplicity, we'll skip for now

def main(args=None):
    rclpy.init(args=args)
    node = DroneTrajectoryController()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()