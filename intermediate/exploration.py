import rclpy
from rclpy.node import Node

from octomap_msgs.msg import Octomap
from octomap_msgs.srv import GetOctomap

import os
import numpy as np

class Exploration(Node):
    def __init__(self):
        super().__init__('explorer')
        self.NUM_DRONES = os.environ.get('NUM_ROBOTS')

def main():
    NUM_DRONES = os.environ.get('NUM_ROBOTS',5)
    print(NUM_DRONES)
    return 0