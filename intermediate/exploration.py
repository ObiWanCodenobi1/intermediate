import rclpy
from rclpy.node import Node

from octomap_msgs.msg import Octomap
from octomap_msgs.srv import GetOctomap
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

import os
import numpy as np
import math
import heapq
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

class Exploration(Node):
    def __init__(self):
        super().__init__('explorer')
        self.agv_pose_sub = self.create_subscription(Point, '/AGV/pose', self.agv_pose_callback,10)

        self.NUM_DRONES = int(os.environ.get('NUM_ROBOTS', 10))
        self.MAX_COMM_DIST = 40
        self.SENSOR_RADIUS = 3
        self.ocgrid3d = np.load('oc_grid.npy')
        self.HEIGHT = 3
        _,_,self.HEIGHT_IDX = self.world2grid(0,0,self.HEIGHT)
        self.ocgrid = self.ocgrid3d[:,:,self.HEIGHT_IDX]
        self.shape = self.ocgrid.shape
        self.res = 0.15

        # Constants from simulator
        self.OBSTACLE_RADIUS = 4  # Adjust based on grid resolution
        self.GRID_CELLS = self.shape[0]  # Assuming square
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
        self.visited_penalty = np.zeros(self.shape)
        self.exploration_reward = np.zeros(self.shape)
        self.exploration_map = np.zeros(self.shape)

        # Grid coord cache
        self.xx, self.yy = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))

        self.cf_poses = {
            f"cf_{i}": {
                "position": {"x": None, "y": None, "z": None},
                "orientation": {"x": None, "y": None, "z": None, "w": None}
            }
            for i in range(1, self.NUM_DRONES+1)
        }
        
        self.global_pose_subscriptions = []
        for i in range(1, self.NUM_DRONES+1):
            self.global_pose_subscriptions.append(self.create_subscription(PoseStamped, f"/cf_{i}/pose", lambda msg, drone_id=i: self.global_pose_callback(msg, drone_id), 10))

        # Publishers for goals
        self.goal_publishers = [self.create_publisher(PoseStamped, f'/cf_{i}/goal', 10) for i in range(1, self.NUM_DRONES+1)]

        self.selected_points = []
        p1x, p1y, _ = self.world2grid(-5,-5,0)
        self.p1 = (p1x, p1y)
        self.p2 = None
        self.drone_positions = [(0, 0) for _ in range(self.NUM_DRONES)]

        # Setup environment
        self._setup_environment()

        # Timer for periodic updates
        self.timer = self.create_timer(0.1, self.step)  # 10 Hz

            
    def grid2world(self, x, y,z):
        return ((x*self.res - 66.5),(y*self.res - 66.5),(z*self.res + 0.5))
    
    def world2grid(self, x, y,z):
        return (int((x + 66.5)/self.res), int((y + 66.5)/self.res), int((y - 0.5)/self.res))
    
    def agv_pose_callback(self, msg):
        p2 = (msg.x,msg.y)

    def global_pose_callback(self,msg:PoseStamped,drone_id):
        self.cf_poses[f"cf_{drone_id}"]["position"]["x"] = msg.pose.position.x
        self.cf_poses[f"cf_{drone_id}"]["position"]["y"] = msg.pose.position.y
        self.cf_poses[f"cf_{drone_id}"]["position"]["z"] = msg.pose.position.z

        self.cf_poses[f"cf_{drone_id}"]["orientation"]["x"] = msg.pose.orientation.x
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["y"] = msg.pose.orientation.y
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["z"] = msg.pose.orientation.z
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["w"] = msg.pose.orientation.w

        grid_x, grid_y, _ = self.world2grid(msg.pose.position.x,msg.pose.position.y,msg.pose.position.z)
        self.drone_positions[drone_id] = [grid_x, grid_y]

def main():
    
    return 0