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

class Frontier_Detection(Node):
    def __init__(self):
        super().__init__('drone_trajectory_controller')

        self.NUM_DRONES = int(os.environ.get('NUM_ROBOTS', 10))
        self.SENSOR_RADIUS = 6
        self.FOV = 60 #field of view in degrees
        ocgrid_path = os.path.expanduser('~/ros2_ws/src/intermediate/intermediate/oc_grid.npy')
        self.ocgrid3d = np.load(ocgrid_path)
        self.HEIGHT = 3
        _, _, self.HEIGHT_IDX = self.world2grid(0, 0, self.HEIGHT)
        self.ocgrid = self.ocgrid3d[:, :, self.HEIGHT_IDX]
        self.res = 0.15

        self.cf_poses = {
            f"cf_{i}": {
                "x":None, "y":None, "yaw":None
            }
            for i in range(1, self.NUM_DRONES + 1)
        }

        self.global_pose_subscriptions = []
        for i in range(1, self.NUM_DRONES + 1):
            self.global_pose_subscriptions.append(self.create_subscription(PoseStamped, f"/cf_{i}/pose", lambda msg, drone_id=i: self.global_pose_callback(msg, drone_id), 10))

        self.target_publisher = self.create_publisher(Targets, 'targets', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def grid2world(self, x, y, z):
        return ((x * self.res) - 66.5, (y * self.res) - 66.5, (z * self.res) + 0.5)

    def world2grid(self, x, y, z):
        #self.res = 0.15  # Assuming resolution
        return (int((x + 66.5) / self.res), int((y + 66.5) / self.res), int((z - 0.5) / self.res))
    
    def global_pose_callback(self, msg: PoseStamped, drone_id):

        q_x = msg.pose.orientation.x
        q_y = msg.pose.orientation.y
        q_z = msg.pose.orientation.z
        q_w = msg.pose.orientation.w
        yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))*(180/math.pi)

        grid_x, grid_y, _ = self.world2grid(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        
        self.cf_poses[f"cf_{drone_id}"]["x"] = grid_x
        self.cf_poses[f"cf_{drone_id}"]["y"] = grid_y
        self.cf_poses[f"cf_{drone_id}"]["yaw"] = yaw

    def update_grid(self):
        for drone_id in range(1, self.NUM_DRONES + 1):
            x0 = self.cf_poses[f"cf_{drone_id}"]["x"]
            y0 = self.cf_poses[f"cf_{drone_id}"]["y"]
            yaw = self.cf_poses[f"cf_{drone_id}"]["yaw"]
            for theta in range(-1*self.fov/2,self.fov/2):
                for d in range(self.SENSOR_RADIUS):
                    x = x0 + int(d*math.cos(math.radians(yaw+theta)))
                    y = y0 + int(d*math.sin(math.radians(yaw+theta)))
                    if(self.ocgrid[x][y]==0):
                        self.ocgrid[x][y]=2
                    if(self.ocgrid[x][y]==1):
                        break

    def get_frontier(self,occupancy_grid, unknown_val=0, free_val=2):
        # 1. Create a binary mask of unknown space
        unknown_mask = (occupancy_grid == unknown_val)
        
        # 2. Create a binary mask of free space
        free_mask = (occupancy_grid == free_val)
        
        # 3. Dilate the unknown mask by 1 pixel
        # This identifies all pixels that are neighbors to unknown cells
        dilated_unknown = binary_dilation(unknown_mask)
        
        # 4. Frontiers are pixels that are FREE and touch UNKNOWN
        frontier_mask = dilated_unknown & free_mask
        
        # Return the (y, x) coordinates of frontier pixels
        return np.argwhere(frontier_mask)
    
    def cluster_frontiers(self,frontier_mask, min_size=3):
        """
        Clusters contiguous frontier pixels into groups.
        Returns a list of dictionaries containing centroids and sizes.
        """
        # Label connected components (clusters)
        # structure=np.ones((3,3)) allows diagonal connections
        labeled_array, num_features = label(frontier_mask, structure=np.ones((3, 3)))
        
        clusters = []
        
        if num_features > 0:
            # Calculate centroids for all clusters at once
            centroids = center_of_mass(frontier_mask, labeled_array, range(1, num_features + 1))
            
            # Calculate sizes (pixel counts)
            # Using bincount is faster than looping
            sizes = np.bincount(labeled_array.ravel())[1:] # Skip background (0)
            
            for i in range(num_features):
                if sizes[i] >= min_size:
                    x_c,y_c,_= self.grid2world(centroids[i][1],centroids[i][0],0)# (y, x) format
                    clusters.append({
                        'id': i + 1,
                        'centroid': (x_c,y_c), 
                        'size': sizes[i]
                    })
                    
        return clusters, labeled_array
    
    def timer_callback(self):
        self.update_grid()
        frontier_mask = self.get_frontier(self.ocgrid)
        clusters, _ = self.cluster_frontiers(frontier_mask)

        msg = Targets()
        targets = []
        for cluster in clusters:
            target = Target()
            target.id = cluster['id']
            target.x = cluster['centroid'][0]
            target.y = cluster['centroid'][1]
            targets.append(target)

        msg.targets = targets
        self.target_publisher.publish(msg)



def main(args=None):
    rclpy.init(args=args)

    frontier_detection = Frontier_Detection()

    rclpy.spin(frontier_detection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    frontier_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()