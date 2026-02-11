import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose,PoseStamped
from custom_interfaces.msg import Target, Targets
import numpy as np
import os
import math
from scipy.ndimage import binary_dilation, label, center_of_mass