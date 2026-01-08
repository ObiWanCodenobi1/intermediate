import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose
import numpy as np

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            MarkerArray,
            'occupied_cells_vis_array',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self._logger.info("created subscription")
        self.res = 0.15
        self.oc_grid = np.zeros((134,134,27)) # 0 for unseen , 1 for occupied, 2 for seen

    def listener_callback(self, msg):       
        for marker in msg.markers:
            for point in marker.points:
                self.oc_grid[int(point.x/self.res + 66.5),int(point.y/self.res + 66.5),int(point.z/self.res - 0.5)] = 1

        np.save('oc_grid.npy',self.oc_grid)

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