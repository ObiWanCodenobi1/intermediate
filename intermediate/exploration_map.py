import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose,PoseStamped
import numpy as np

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.oc_array_sub = self.create_subscription(
            MarkerArray,
            'occupied_cells_vis_array',
            self.listener_callback,
            10)
        self.oc_array_sub  # prevent unused variable warning
        self._logger.info("created subscription")
        self.res = 0.15
        self.oc_grid = np.zeros((134,134,27)) # 0 for unseen , 1 for occupied, 2 for seen

        self.no_of_drones = 1

        self.cf_poses = {
        f"cf_{i}": {
            "position": {"x": None, "y": None, "z": None},
            "orientation": {"x": None, "y": None, "z": None, "w": None}
        }
        for i in range(1, self.no_of_drones+1)
        }
        
        #for i in range(self.no_of_drones):
        self.global_pose_subscriptions = []
        for i in range(1, self.no_of_drones+1):
            self.global_pose_subscriptions.append(self.create_subscription(PoseStamped, f"/cf_{i}/pose", lambda msg, drone_id=i: self.global_pose_callback(msg, drone_id), 10))

        self.timer = self.create_timer(10, self.timer_callback)

            

    def listener_callback(self, msg):       
        for marker in msg.markers:
            for point in marker.points:
                self.oc_grid[int(point.x/self.res + 66.5),int(point.y/self.res + 66.5),int(point.z/self.res - 0.5)] = 1

        np.save('oc_grid.npy',self.oc_grid)

    def global_pose_callback(self,msg:PoseStamped,drone_id):
        self.cf_poses[f"cf_{drone_id}"]["position"]["x"] = msg.pose.position.x
        self.cf_poses[f"cf_{drone_id}"]["position"]["y"] = msg.pose.position.y
        self.cf_poses[f"cf_{drone_id}"]["position"]["z"] = msg.pose.position.z

        self.cf_poses[f"cf_{drone_id}"]["orientation"]["x"] = msg.pose.orientation.x
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["y"] = msg.pose.orientation.y
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["z"] = msg.pose.orientation.z
        self.cf_poses[f"cf_{drone_id}"]["orientation"]["w"] = msg.pose.orientation.w

    def timer_callback(self):
        print(self.cf_poses)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()