import numpy as np
import cv2
import random
import math
from collections import deque
import time
import heapq

# =============================================================================
# ICUAS CV2 Simulator V4: Connection Recovery
# Implements cascaded relocation logic to restore connectivity.
# =============================================================================

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

class CombinedExploration:
    def __init__(self):
        # Constants
        self.LOGICAL_SIZE = 200
        self.DISPLAY_SIZE = 400
        self.SCALE_FACTOR = 2

        self.NUM_DRONES = 10
        self.NUM_OBSTACLES = 20
        self.OBSTACLE_RADIUS = 4
        self.MAX_COMM_DIST = 40
        self.SENSOR_RADIUS = 3

        self.GRID_CELLS = self.LOGICAL_SIZE

        # Force field / height map parameters
        self.SIGMA = 3.0
        self.GROUND_PLANE = 0.0
        self.FILL_INCREMENT = 1.0
        self.G_REDUCTION_FACTOR = 0.1
        self.G_GAIN = 8.0
        self.MAX_FORCE_STEP = 3
        self.ZERO_RADIUS = 1
        self.EPS = 1e-3
        self.KEY_STEP = 1
        self.T_REDUCE = 0.90
        self.T_ZERO = 0.95
        self.ALPHA = 0.1

        # Restoration Constants
        self.MAX_RELOCATION_STEPS = 100
        self.RESTORATION_COOLDOWN_TICKS = 20
        self.RESTORATION_RECURSION_LIMIT = 5
        self.MAX_MOVE_PER_DRONE = 3
        self.PDR_TRIGGER_THRESHOLD = 0.9 * self.MAX_COMM_DIST # Proactive trigger

        # Phase Control
        self.phase = "DEPLOYMENT"
        self.convergence_counter = 0
        self.prev_score = 0
        self.restoration_active = False
        self.restoration_cooldown = 0
        self.relocation_plan = [] # List of (drone_idx, target_pos) for viz

        # Environment
        self.obstacles = []
        self.los_cache = {}

        # Exploration Maps
        self.visited_penalty = np.zeros((self.GRID_CELLS, self.GRID_CELLS))
        self.exploration_reward = np.zeros((self.GRID_CELLS, self.GRID_CELLS))
        self.exploration_map = np.zeros((self.GRID_CELLS, self.GRID_CELLS))

        # Force-field data
        self.g_masks = None
        self.gauss_stack = None
        self.total_cells = None
        self.visited_cells = None
        self.obstacle_grid = None
        self.height_map = None
        self.g_field_magnitude = None

        # Grid coord cache
        grid_range = np.arange(self.GRID_CELLS)
        self.xx, self.yy = np.meshgrid(grid_range, grid_range)

        # State
        self.selected_points = []
        self.p1 = None
        self.p2 = None
        self.drone_positions = []

        # Visualization
        self.main_image = np.zeros((self.DISPLAY_SIZE, self.DISPLAY_SIZE, 3), dtype=np.uint8)
        self.exploration_image = np.zeros((self.DISPLAY_SIZE, self.DISPLAY_SIZE, 3), dtype=np.uint8)

        self._setup_environment()
        self.g_field_magnitude = self._compute_g_field()

    def _setup_environment(self):
        print("Setting up environment...")
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            attempts = 0
            while attempts < 100:
                x = random.randint(self.OBSTACLE_RADIUS, self.LOGICAL_SIZE - self.OBSTACLE_RADIUS)
                y = random.randint(self.OBSTACLE_RADIUS, self.LOGICAL_SIZE - self.OBSTACLE_RADIUS)
                too_close = False
                for obs_x, obs_y in self.obstacles:
                    if math.hypot(x - obs_x, y - obs_y) < self.OBSTACLE_RADIUS * 3:
                        too_close = True; break
                if not too_close:
                    self.obstacles.append((x, y))
                    break
                attempts += 1

        gauss_list = []
        self.g_masks = np.ones((self.NUM_OBSTACLES, self.GRID_CELLS, self.GRID_CELLS), dtype=float)
        self.total_cells = np.zeros(self.NUM_OBSTACLES, dtype=float)
        self.visited_cells = np.zeros(self.NUM_OBSTACLES, dtype=float)

        eps_mask = 1e-6
        # Precompute boolean obstacle grid for fast A* lookup
        self.obstacle_grid = np.zeros((self.GRID_CELLS, self.GRID_CELLS), dtype=bool)
        for x in range(self.GRID_CELLS):
            for y in range(self.GRID_CELLS):
                blocked = False
                for ox, oy in self.obstacles:
                    if math.hypot(x - ox, y - oy) <= self.OBSTACLE_RADIUS + 2:
                        blocked = True; break
                self.obstacle_grid[y, x] = blocked

        for idx, (obs_x, obs_y) in enumerate(self.obstacles):
            dx = self.xx - obs_x
            dy = self.yy - obs_y
            dist = np.hypot(dx, dy)
            pit = np.exp(-(dist ** 2) / (2 * self.SIGMA ** 2))
            
            # User requirement: 90% of "surrounding cells" (ring of radius 3)
            # Exclude interior of obstacle
            ring_mask = (dist > self.OBSTACLE_RADIUS) & (dist <= self.OBSTACLE_RADIUS + 3)
            
            # Influence is now strictly this ring
            influence = ring_mask 
            # We keep the Gaussian values for height map, but "influence" for coverage count
            # is just the ring.
            
            gauss_list.append(pit)
            # For the masks, we initialize them to 1.0 where we care (the ring)
            # Outside the ring, we don't care about counting them for "coverage percentage"
            # BUT we do care about the G-field force existing further out.
            # The user says "reducing the g effect after 90% coverage".
            # So `total_cells` should count the ring. `visited_cells` counts the ring.
            
            self.total_cells[idx] = float(np.count_nonzero(ring_mask))
            
            # We initialize the mask. The mask multiplies the Gaussian force.
            # Initially 1.0 everywhere.
            # When visited, we set mask to 0.0. 
            
            # Wait, if we only count the ring for the percentage, we should likely
            # only zero-out the ring in the mask? 
            # Or does visiting the ring kill the force for the whole obstacle?
            # "Reducing the g effect... if 90% of cells... is visited"
            # Implies the trigger is the ring coverage. The effect is global reduction.
            
        self.gauss_stack = np.stack(gauss_list, axis=0)
        self.height_map = -np.sum(self.gauss_stack * self.g_masks, axis=0)
        print("Environment setup complete.")

        print("Environment setup complete.")

    def _point_line_distance(self, px, py, x1, y1, x2, y2):
        line_mag = math.hypot(x2 - x1, y2 - y1)
        if line_mag < 1e-6:
            return math.hypot(px - x1, py - y1)
        u = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)))
        return math.hypot(px - (x1 + u * (x2 - x1)), py - (y1 + u * (y2 - y1)))

    def _has_los(self, pos1, pos2):
        key = (min(pos1, pos2), max(pos1, pos2))
        if key in self.los_cache:
            return self.los_cache[key]

        if math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]) > self.MAX_COMM_DIST:
            self.los_cache[key] = False; return False

        for ox, oy in self.obstacles:
            if self._point_line_distance(ox, oy, pos1[0], pos1[1], pos2[0], pos2[1]) <= self.OBSTACLE_RADIUS + 1:
                self.los_cache[key] = False; return False

        self.los_cache[key] = True; return True

    def _is_valid(self, pos):
        if not (0 <= pos[0] < self.GRID_CELLS and 0 <= pos[1] < self.GRID_CELLS): return False
        if self.obstacle_grid is not None: return not self.obstacle_grid[pos[1], pos[0]]
        for ox, oy in self.obstacles:
            if math.hypot(pos[0]-ox, pos[1]-oy) <= self.OBSTACLE_RADIUS + 2: return False
        return True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 2:
                lx = int(x // self.SCALE_FACTOR); ly = int(y // self.SCALE_FACTOR)
                lx = max(0, min(self.GRID_CELLS-1, lx)); ly = max(0, min(self.GRID_CELLS-1, ly))
                self.selected_points.append((lx, ly))
                print(f"Point {len(self.selected_points)}: {(lx, ly)}")

    def wait_for_selection(self):
        print("Click 2 points to define initial deployment area...")
        cv2.namedWindow('Simulation')
        cv2.setMouseCallback('Simulation', self.on_mouse)
        while len(self.selected_points) < 2:
            self.main_image.fill(0)
            for obs in self.obstacles:
                cv2.circle(self.main_image, self._logical_to_display(obs), self.OBSTACLE_RADIUS * self.SCALE_FACTOR, (0, 0, 255), -1)
            for i, p in enumerate(self.selected_points):
                pt = self._logical_to_display(p)
                cv2.circle(self.main_image, pt, 6, (0, 255, 255), -1)
                cv2.putText(self.main_image, f"P{i+1}", (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow('Simulation', self.main_image)
            if cv2.waitKey(50) & 0xFF == ord('q'): exit()
        self.p1 = self.selected_points[0]
        self.p2 = self.selected_points[1]

        # Init drones
        start, end = self.p1, self.p2
        for i in range(1, self.NUM_DRONES + 1):
            t = i / (self.NUM_DRONES + 1)
            x = int(start[0] + t * (end[0] - start[0]))
            y = int(start[1] + t * (end[1] - start[1]))
            if not self._is_valid((x, y)):
                for r in range(1, 10):
                    found = False
                    for ang in np.linspace(0, 6.28, 8):
                        nx = int(x + r * math.cos(ang)); ny = int(y + r * math.sin(ang))
                        if self._is_valid((nx, ny)): x, y = nx, ny; found = True; break
                    if found: break
            self.drone_positions.append((x, y))

    def _get_connected_components(self, positions):
        nodes = positions + [self.p1, self.p2]
        n = len(nodes)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i+1, n):
                if self._has_los(nodes[i], nodes[j]):
                    uf.union(i, j)
        groups = {}
        for i in range(n):
            root = uf.find(i)
            if root not in groups: groups[root] = []
            groups[root].append(i)
        return groups

    # =========================================================================
    # RESTORATION LOGIC START
    # =========================================================================

    def enter_restoration_mode(self):
        if not self.restoration_active:
            self.restoration_active = True
            print(">>> CRITICAL LOSS DETECTED: Entering RESTORATION mode <<<")

    def exit_restoration_mode(self):
        if self.restoration_active:
            self.restoration_active = False
            self.restoration_cooldown = self.RESTORATION_COOLDOWN_TICKS
            self.relocation_plan = []
            print(">>> Connectivity Restored: Resuming EXPLORATION <<<")

    def _solve_cascade(self, idx, target_pos, current_positions, moved_indices, depth=0):
        # Base case
        if depth > self.RESTORATION_RECURSION_LIMIT:
            return []

        # Proposal for idx
        if idx in moved_indices: return [] # Cycle avoidance
        
        # Check simple move
        curr_pos = current_positions[idx]
        delta_x = target_pos[0] - curr_pos[0]
        delta_y = target_pos[1] - curr_pos[1]
        dist = math.hypot(delta_x, delta_y)

        step = min(self.MAX_MOVE_PER_DRONE, dist)
        if step < 1.0: return []
        
        nx = int(curr_pos[0] + (delta_x / dist) * step)
        ny = int(curr_pos[1] + (delta_y / dist) * step)

        if not self._is_valid((nx, ny)):
            return [] # Blocked transform

        # Collision check with static obstacles done by is_valid
        # Collision check with other drones? (Simplified: check against current_positions)
        for i, pos in enumerate(current_positions):
            if i != idx and i not in moved_indices:
                if pos == (nx, ny): return [] # Collision 

        # Now the Elastic Band Check: Does this move break links with *current neighbors*?
        # 1. Identify neighbors of idx in current_positions
        # Simplified: Check against ALL nodes. If link exists now, must usually stay linked or be pulled.
        # Ideally, we only care about neighbors that are NOT in the same target component direction.
        
        nodes = current_positions + [self.p1, self.p2]
        my_neighbors = []
        for i, pos_i in enumerate(nodes):
            if i == idx: continue
            if self._has_los(nodes[idx], pos_i):
                my_neighbors.append(i)

        proposed_moves = [(idx, (nx, ny))]
        temp_moved_indices = moved_indices | {idx}
        
        # Check connectivity of neighbors to new pos
        # If a neighbor loses connection, we must recurse and pull them
        for n_idx in my_neighbors:
            # P1 and P2 cannot move
            pos_n = nodes[n_idx]
            # Check LOS to new pos of idx
            if not self._has_los((nx, ny), pos_n):
                # Link broken!
                if n_idx >= self.NUM_DRONES:
                    return [] # Cannot pull P1/P2, abort move responsible for break
                
                # Must pull n_idx towards (nx, ny)
                cascade = self._solve_cascade(n_idx, (nx, ny), current_positions, temp_moved_indices, depth + 1)
                if not cascade:
                    return [] # Cascade failed, abort entire branch
                proposed_moves.extend(cascade)
                for m_idx, _ in cascade:
                    temp_moved_indices.add(m_idx)
        
        return proposed_moves

    def _heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _get_path_a_star(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        count = 0 
        MAX_COUNT = 5000 
        found = False
        current = start

        while frontier:
            count += 1
            if count > MAX_COUNT: break 
            _, current = heapq.heappop(frontier)

            if math.hypot(current[0]-goal[0], current[1]-goal[1]) < 2.0:
                found = True; break
            
            x, y = current
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
                 nx, ny = x + dx, y + dy
                 if not (0 <= nx < self.GRID_CELLS and 0 <= ny < self.GRID_CELLS): continue
                 if not self._is_valid((nx, ny)): continue

                 new_cost = cost_so_far[current] + math.hypot(dx, dy)
                 next_node = (nx, ny)
                 
                 if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                     cost_so_far[next_node] = new_cost
                     priority = new_cost + self._heuristic(next_node, goal)
                     heapq.heappush(frontier, (priority, next_node))
                     came_from[next_node] = current
        
        path = []
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None: break
        path.reverse()
        return path

    def _plan_relocation_step(self):
        # 1. Build component interaction graph
        groups = self._get_connected_components(self.drone_positions)
        if len(groups) <= 1:
            return None # Resolved
        
        group_ids = list(groups.keys())
        best_pair_val = float('inf')
        best_merge_move = None 
        nodes = self.drone_positions + [self.p1, self.p2]

        for i in range(len(group_ids)):
            for j in range(i+1, len(group_ids)):
                g1 = groups[group_ids[i]]
                g2 = groups[group_ids[j]]

                for u in g1:
                    pos_u = nodes[u]
                    for v in g2:
                        pos_v = nodes[v]
                        dist = math.hypot(pos_u[0]-pos_v[0], pos_u[1]-pos_v[1])
                        cost = dist 
                        
                        if cost >= best_pair_val: continue

                        # Try move U -> V (if U is drone)
                        if u < self.NUM_DRONES:
                            path = self._get_path_a_star(pos_u, pos_v)
                            if path and len(path) > 1:
                                # Crawl along path
                                target_wp = path[1]
                                accum = 0; last = path[1]
                                for pt in path[1:]:
                                    accum += math.hypot(pt[0]-last[0], pt[1]-last[1])
                                    if accum >= self.MAX_MOVE_PER_DRONE:
                                        target_wp = pt; break
                                    last = pt; target_wp = pt
                                
                                moves = self._solve_cascade(u, target_wp, self.drone_positions, set())
                                if moves:
                                    best_pair_val = cost
                                    best_merge_move = moves

                        # Try move V -> U (if V is drone)
                        if v < self.NUM_DRONES:
                            path = self._get_path_a_star(pos_v, pos_u)
                            if path and len(path) > 1:
                                # Crawl along path
                                target_wp = path[1]
                                accum = 0; last = path[1]
                                for pt in path[1:]:
                                    accum += math.hypot(pt[0]-last[0], pt[1]-last[1])
                                    if accum >= self.MAX_MOVE_PER_DRONE:
                                        target_wp = pt; break
                                    last = pt; target_wp = pt

                                moves = self._solve_cascade(v, target_wp, self.drone_positions, set())
                                if moves:
                                    if cost < best_pair_val: 
                                        best_pair_val = cost
                                        best_merge_move = moves

        return best_merge_move

    # =========================================================================
    # RESTORATION LOGIC END
    # =========================================================================

    def _calculate_deployment_score(self, positions):
        groups = self._get_connected_components(positions)
        if len(groups) > 1: return -1e6 * len(groups)
        score = 0; step = 5; r_sq = self.SENSOR_RADIUS ** 2
        all_sensors = positions + [self.p1, self.p2]
        for x in range(0, self.GRID_CELLS, step):
            for y in range(0, self.GRID_CELLS, step):
                for sx, sy in all_sensors:
                    if (x-sx)**2 + (y-sy)**2 <= r_sq: score += 1; break
        score -= self._comm_soft_penalty(positions)
        return score

    def _calculate_exploration_score(self, positions):
        groups = self._get_connected_components(positions)
        if len(groups) > 1: return -1e6 * len(groups)
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
            nearest = None
            for j, m in enumerate(nodes):
                if i == j: continue
                d = math.hypot(n[0]-m[0], n[1]-m[1])
                if d <= self.MAX_COMM_DIST and (nearest is None or d < nearest): nearest = d
            if nearest is None: penalty += 1e4
            elif nearest > safety_thresh: penalty += (nearest - safety_thresh) * 50.0
        return penalty

    def _apply_visit_effects(self, positions):
        eps_mask = 1e-6
        for pos in positions:
            self.visited_penalty[pos[1]][pos[0]] += 0.2
            r = self.ZERO_RADIUS
            x_min = max(0, pos[0]-r); x_max = min(self.GRID_CELLS, pos[0]+r+1)
            y_min = max(0, pos[1]-r); y_max = min(self.GRID_CELLS, pos[1]+r+1)
            self.exploration_map[y_min:y_max, x_min:x_max] = 1
            
            # Slice global coords for this patch
            local_xx = self.xx[y_min:y_max, x_min:x_max]
            local_yy = self.yy[y_min:y_max, x_min:x_max]

            for i in range(self.NUM_OBSTACLES):
                # Calculate dist relative to this obstacle
                dx = local_xx - self.obstacles[i][0]
                dy = local_yy - self.obstacles[i][1]
                local_dist = np.hypot(dx, dy)
                
                # Identify the "Ring" of interest (R < d <= R+3)
                ring_mask_local = (local_dist > self.OBSTACLE_RADIUS) & (local_dist <= self.OBSTACLE_RADIUS + 3)
                
                # Check for new visits within the Ring
                current_mask = self.g_masks[i, y_min:y_max, x_min:x_max]
                
                # Overlap of [Still Active Force] AND [Is In Ring]
                newly_visited_ring = (current_mask > 0) & ring_mask_local
                count_new = np.count_nonzero(newly_visited_ring)
                
                if count_new > 0:
                     self.visited_cells[i] += float(count_new)
                
                # Clear force in the visited patch (stops attraction to here)
                self.g_masks[i, y_min:y_max, x_min:x_max] = 0.0
                     
        self.height_map = -np.sum(self.gauss_stack * self.g_masks, axis=0)

    def _logical_to_display(self, pos):
        return (pos[0]*self.SCALE_FACTOR, pos[1]*self.SCALE_FACTOR)

    def _compute_per_obstacle_fraction(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            f = np.divide(self.visited_cells, self.total_cells, out=np.zeros_like(self.visited_cells), where=self.total_cells>0)
        return np.clip(f, 0.0, 1.0)

    def _g_scale_for_fraction(self, f):
        if f >= self.T_ZERO: return 0.01
        if f >= self.T_REDUCE: return 0.10
        return 1.0

    def _compute_force_vector(self, pos):
        fx = 0.0; fy = 0.0
        f_obs = self._compute_per_obstacle_fraction()
        for idx, (ox, oy) in enumerate(self.obstacles):
            mask_val = self.g_masks[idx, pos[1], pos[0]]
            if mask_val <= 0: continue
            scale = self._g_scale_for_fraction(f_obs[idx])
            if scale <= 0: continue
            dx = ox - pos[0]; dy = oy - pos[1]
            dist = math.hypot(dx, dy) + self.EPS
            mag = self.G_GAIN * scale * (1.0 / dist)
            fx += (dx/dist)*mag*mask_val; fy += (dy/dist)*mag*mask_val
        return fx, fy

    def _compute_g_field(self):
        f_obs = self._compute_per_obstacle_fraction()
        g_field = np.zeros_like(self.height_map)
        for idx, (ox, oy) in enumerate(self.obstacles):
            scale = self._g_scale_for_fraction(f_obs[idx])
            if scale <= 0: continue
            dx = ox - self.xx; dy = oy - self.yy
            dist = np.hypot(dx, dy) + self.EPS
            contrib = self.G_GAIN * scale * (1.0 / dist) * self.g_masks[idx]
            g_field += contrib
        g_field = np.nan_to_num(g_field, nan=0.0, posinf=0.0, neginf=0.0)
        return g_field

    def _move_p2(self, dx, dy):
        nx = int(np.clip(self.p2[0]+dx, 0, self.GRID_CELLS-1))
        ny = int(np.clip(self.p2[1]+dy, 0, self.GRID_CELLS-1))
        if self._is_valid((nx, ny)): self.p2 = (nx, ny)

    def _is_collision_free(self, positions):
        return len(positions) == len(set(positions))

    def _add_force_candidate(self, candidates, curr_pos, idx):
        fx, fy = self._compute_force_vector(curr_pos[idx])
        norm = math.hypot(fx, fy)
        if norm < 1e-6: return
        dir_x = fx/norm; dir_y = fy/norm
        step_len = min(self.MAX_FORCE_STEP, max(1.0, norm))
        for _ in range(3):
            nx = int(round(curr_pos[idx][0] + dir_x * step_len))
            ny = int(round(curr_pos[idx][1] + dir_y * step_len))
            if self._is_valid((nx, ny)):
                new_state = curr_pos[:]
                new_state[idx] = (nx, ny)
                if self._is_collision_free(new_state):
                    candidates.append(new_state); return
            step_len = max(1.0, step_len * 0.5)

    def _draw(self):
        self.main_image.fill(0)
        self.exploration_image.fill(0)
        for obs in self.obstacles:
            cv2.circle(self.main_image, self._logical_to_display(obs), self.OBSTACLE_RADIUS*self.SCALE_FACTOR, (0,0,255), -1)
        
        cv2.circle(self.main_image, self._logical_to_display(self.p1), 8, (0,255,255), -1)
        cv2.circle(self.main_image, self._logical_to_display(self.p2), 6, (0,255,255), -1)
        cv2.putText(self.main_image, "P1", (self._logical_to_display(self.p1)[0], self._logical_to_display(self.p1)[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(self.main_image, "P2", (self._logical_to_display(self.p2)[0], self._logical_to_display(self.p2)[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        nodes = self.drone_positions + [self.p1, self.p2]
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if self._has_los(nodes[i], nodes[j]):
                    d = math.hypot(nodes[i][0]-nodes[j][0], nodes[i][1]-nodes[j][1])
                    edges.append((d, i, j))
        edges.sort()
        uf = UnionFind(len(nodes))
        for w, u, v in edges:
            if uf.union(u, v):
                p1 = self._logical_to_display(nodes[u])
                p2 = self._logical_to_display(nodes[v])
                cv2.line(self.main_image, p1, p2, (0,255,0), 2)
        
        # Restoration Plan Visualization
        if self.restoration_active and self.relocation_plan:
             for idx, (tx, ty) in self.relocation_plan:
                 p_start = self._logical_to_display(self.drone_positions[idx])
                 p_end = self._logical_to_display((tx, ty))
                 cv2.arrowedLine(self.main_image, p_start, p_end, (0, 0, 255), 2, tipLength=0.3)

        for i, pos in enumerate(self.drone_positions):
            pt = self._logical_to_display(pos)
            color = (0, 0, 255) if self.restoration_active else ((200,200,200) if self.phase == "DEPLOYMENT" else (50, 200, 255))
            cv2.circle(self.main_image, pt, 4, color, -1)
            cv2.circle(self.main_image, pt, self.SENSOR_RADIUS*self.SCALE_FACTOR, (30,30,30), 1)

        exp_map_vis = cv2.resize(self.exploration_map, (self.DISPLAY_SIZE, self.DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        self.exploration_image = (exp_map_vis * 255).astype(np.uint8)
        self.exploration_image = cv2.cvtColor(self.exploration_image, cv2.COLOR_GRAY2BGR)
        for obs in self.obstacles:
            cv2.circle(self.exploration_image, self._logical_to_display(obs), self.OBSTACLE_RADIUS*self.SCALE_FACTOR, (0,0,255), -1)

        # HUD
        mode_str = "RESTORING..." if self.restoration_active else self.phase
        cv2.putText(self.main_image, f"MODE: {mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255) if not self.restoration_active else (0, 0, 255), 2)
        
        cv2.imshow('Main View', self.main_image)
        cv2.imshow('Exploration', self.exploration_image)
        
        self.g_field_magnitude = self._compute_g_field()
        g_log = np.log1p(np.nan_to_num(self.g_field_magnitude))
        g_norm = cv2.normalize(g_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        g_vis = cv2.resize(g_norm.astype(np.uint8), (self.DISPLAY_SIZE, self.DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        g_vis = cv2.applyColorMap(g_vis, cv2.COLORMAP_JET)
        cv2.imshow('G Field', g_vis)

        h_norm = cv2.normalize(np.nan_to_num(self.height_map), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        h_vis = cv2.resize(h_norm.astype(np.uint8), (self.DISPLAY_SIZE, self.DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        h_vis = cv2.applyColorMap(h_vis, cv2.COLORMAP_JET)
        cv2.imshow('Height Map', h_vis)

    # =========================================================================
    # TESTS
    # =========================================================================
    
    def run_tests(self):
        print("\n--- Running Unit Tests ---")
        
        # Test A: Broken Link Detection
        print("[Test A] Disconnect Scenario:")
        self.drone_positions = [(20, 20), (180, 180)] + [(20, 20)] * (self.NUM_DRONES-2) # Split
        groups = self._get_connected_components(self.drone_positions)
        print(f"  Nodes: {len(self.drone_positions)+2}, Groups: {len(groups)}")
        if len(groups) > 1: print("  PASS: Multiple components detected.")
        else: print("  FAIL: Should assume disconnected.")
        
        # Test B: Cascade Proposal
        print("[Test B] Planning Relocation:")
        plan = self._plan_relocation_step()
        if plan: 
            print(f"  PASS: Plan proposed {len(plan)} moves.")
            for i, p in plan: print(f"    Drone {i} -> {p}")
        else: print("  FAIL: No plan.")
        
        # Restore for run
        self.drone_positions = []
        # Re-init
        start, end = self.p1, self.p2
        for i in range(1, self.NUM_DRONES + 1):
            t = i / (self.NUM_DRONES + 1)
            x = int(start[0] + t * (end[0] - start[0]))
            y = int(start[1] + t * (end[1] - start[1]))
            # Simplistic re-init
            self.drone_positions.append((x, y))
        print("--- Tests Complete ---\n")

    def run(self):
        self.wait_for_selection()
        self.run_tests() # Run tests once at start

        while True:
            # 1. Connection Monitoring
            groups = self._get_connected_components(self.drone_positions)
            
            if len(groups) > 1:
                self.enter_restoration_mode()
            else:
                if self.restoration_active:
                    # Hysteresis / Cooldown
                    if self.restoration_cooldown > 0:
                        self.restoration_cooldown -= 1
                    else:
                        self.exit_restoration_mode()

            # 2. Logic Branch
            if self.restoration_active:
                # Restoration Logic
                moves = self._plan_relocation_step()
                if moves:
                    self.relocation_plan = moves
                    for idx, target in moves:
                        # Apply moves directly (single step execution)
                        # Ensure no collision with other proposed moves
                        self.drone_positions[idx] = target
                else:
                    self.relocation_plan = []
                    # Stuck or no valid move found? 
                    # Jiggle?
                    pass
                
                # No scoring update or convergence check in restoration
                current_score = -1e6 # dummy

            else:
                # Normal Exploration Logic
                candidates = []
                curr_pos = self.drone_positions
                # Random perturbations
                for i in range(self.NUM_DRONES):
                    for _ in range(5):
                        dx = random.randint(-4, 4); dy = random.randint(-4, 4)
                        nx = curr_pos[i][0] + dx; ny = curr_pos[i][1] + dy
                        if not self._is_valid((nx, ny)): continue
                        new_state = curr_pos[:]
                        new_state[i] = (nx, ny)
                        if self._is_collision_free(new_state): candidates.append(new_state)

                # Force candidates
                for i in range(self.NUM_DRONES):
                    self._add_force_candidate(candidates, curr_pos, i)

                if self.phase == "DEPLOYMENT":
                    current_score = self._calculate_deployment_score(curr_pos)
                else:
                    current_score = self._calculate_exploration_score(curr_pos)

                best_score = -float('inf')
                best_state = None

                for s in candidates:
                    if self.phase == "DEPLOYMENT": sc = self._calculate_deployment_score(s)
                    else: sc = self._calculate_exploration_score(s)
                    if sc > best_score: best_score = sc; best_state = s

                if best_state is not None and best_score >= current_score:
                    self.drone_positions = best_state
                    current_score = best_score
                
                # Phase switching
                if self.phase == "DEPLOYMENT":
                    if current_score > 0 and self.prev_score > 0:
                        gain = (current_score - self.prev_score) / self.prev_score
                        if gain < 0.005: self.convergence_counter += 1
                        else: self.convergence_counter = 0
                        if self.convergence_counter >= 3:
                            self.phase = "EXPLORATION"
                            self._draw(); cv2.waitKey(2000)
                            self.prev_score = 0; self.convergence_counter = 0
                    self.prev_score = current_score
                elif self.phase == "EXPLORATION":
                    self._apply_visit_effects(self.drone_positions)

            # 3. Visualization & Input
            self._draw()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('w'): self._move_p2(0, -self.KEY_STEP)
            elif key == ord('s'): self._move_p2(0, self.KEY_STEP)
            elif key == ord('a'): self._move_p2(-self.KEY_STEP, 0)
            elif key == ord('d'): self._move_p2(self.KEY_STEP, 0)
            # Manual verify: Break link with 'b'
            elif key == ord('b'):
                # Teleport drone 0 away to break link
                print("DEBUG: Manually breaking link...")
                self.drone_positions[0] = (self.drone_positions[0][0] + 50, self.drone_positions[0][1] + 50) 
                 

if __name__ == "__main__":
    sim = CombinedExploration()
    sim.run()
