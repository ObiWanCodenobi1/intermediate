import os
from crazyflie_interfaces.srv import Takeoff
import rclpy
from rclpy.node import Node


class TakeoffClientAsync(Node):

    def __init__(self):
        super().__init__('takeoff_client_async')
        self.NUM_DRONES = int(os.environ.get('NUM_ROBOTS', 5))
        self.req = Takeoff.Request()
        self.req.height = 3.0
        self.takeoff_clients = {}
        for i in range(1,self.NUM_DRONES+1):
            self.takeoff_clients[i] = self.create_client(Takeoff, f'/cf_{i}/takeoff')
            while not self.takeoff_clients[i].wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'takeoff_service {i} not available, waiting again...')

    def send_request(self):
        for i in range(1,self.NUM_DRONES+1): 
            self.takeoff_clients[i].call_async(self.req)
        


def main():
    rclpy.init()

    minimal_client = TakeoffClientAsync()
    future = minimal_client.send_request()
    rclpy.spin_until_future_complete(minimal_client, future)
    response = future.result()
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()