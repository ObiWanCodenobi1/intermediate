import rclpy
import rclpy.node
from crazyflie_interfaces.srv import GoTo

class Solution(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(GoTo, '/cf_1/go_to')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = GoTo.Request()

    def send_request(self,x,y,z,yaw):
        self.req.relative = False
        self.req.goal.x = x
        self.req.goal.y = y
        self.req.goal.z = z
        return self.cli.call_async(self.req)
    

def main():
    rclpy.init()

    minimal_client = Solution()
    future = minimal_client.send_request(1.0,1.0,1.0)
    rclpy.spin_until_future_complete(minimal_client,future)
    minimal_client.get_logger().info("Going to (1,1,1)")

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()