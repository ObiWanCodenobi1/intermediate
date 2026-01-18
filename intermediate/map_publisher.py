import rclpy
from rclpy.node import Node

from octomap_msgs.msg import Octomap
from octomap_msgs.srv import GetOctomap

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        self.cli = self.create_client(GetOctomap, 'octomap_binary')
        self._logger.info('created client')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = GetOctomap.Request()
        self._logger.info('created request')

        self.pub = self.create_publisher(Octomap,'/octomap',10)
        self._logger.info('created publisher')
        timer_period = 1 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self._logger.info('created timer')
    
    def timer_callback(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('sent request')
        if future.result() and future.result().success:
            result = future.result().message
            map = result.map
            self.pub.publish(map)
            self._logger('published octomap')
        else:
            self.get_logger().error('Failed to call /octomap_binary service or service returned failure.')

def main(args=None):
    rclpy.init(args=args)

    octomap_publisher = MapPublisher()

    rclpy.spin(octomap_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    octomap_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()