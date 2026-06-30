import rclpy
from rclpy.node import Node

from octomap_msgs.msg import Octomap
from octomap_msgs.srv import GetOctomap

import os
import numpy as np
import math
from collections import deque
import time
import heapq

class Exploration(Node):
    def __init__(self):
        super().__init__('explorer')
        self.NUM_DRONES = os.environ.get('NUM_ROBOTS')
        self.oc_grid_3d = np.load("oc_grid.npy")
        self.res = 0.15
        self.height = 3
        _,_,self.height_index = self.world2grid(0,0,self.height)
        self.exploration_map = self.oc_grid_3d[:,:,self.height_index]
        self.visited_penalty = np.zeros(self.exploration_map.shape())
        self.exploration_reward = np.zeros(self.exploration_map.shape())

    def grid2world(self,x,y,z):
        return (((x-66.5)*self.res,(y-66.5)*self.res),(z+0.5)*self.res)
    
    def world2grid(self,x,y,z):
        return ((int((x+66.5)/self.res),int((y+66.5)/self.res)),int((z-0.5)/self.res))

def main():
    NUM_DRONES = os.environ.get('NUM_ROBOTS',5)
    print(NUM_DRONES)
    return 0