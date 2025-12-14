import rclpy
import rclpy.node
from crazyflie_interfaces.srv import Takeoff
from crazyflie_interfaces.srv import GoTo
import time

class Solution(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(Takeoff, '/cf_1/takeoff')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Takeoff.Request()

        self.goto = self.create_client(GoTo, '/cf_1/go_to')
        while not self.goto.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('goto service not available, waiting again...')
        self.coord = GoTo.Request()


    def takeoff_request(self):
        self.req.group_mask = 1
        self.req.height = 2.0
        self.req.duration.sec = 1
        self.req.duration.nanosec = 0
        self.get_logger().info("Crazyflie 1 has taken off") 
        return self.cli.call_async(self.req)
        

    def send_coords(self,x,y,z,yaw):
        self.coord.goal.x = x
        self.coord.goal.y = y
        self.coord.goal.z = z
        self.coord.yaw = yaw
        self.get_logger().info("Crazyflie is going to (%d,%d,%d) with yaw %d degrees" %(x,y,z,yaw))
        return self.goto.call_async(self.coord)
    
def main():
    rclpy.init()

    takeoff = Solution()
    future = takeoff.takeoff_request()
    
    rclpy.spin_until_future_complete(takeoff,future)
    takeoff.destroy_node()
    time.sleep(5)
    go = Solution()
    future2 = go.send_coords(-2.0,-5.0,2.0,0.0)
    rclpy.spin_until_future_complete(go,future2)
    go.destroy_node()
    time.sleep(15)

    go1 = Solution()
    future3 = go1.send_coords(-2.0,-2.0,2.0,0.0)
    rclpy.spin_until_future_complete(go1,future3)
    response = future.result()

    # goto = go()
    # future = goto.send_request(1,1,1)
    # rclpy.spin_until_future_complete(goto, future)
    # response = future.result()
    # goto.get_logger().info("Crazyflie 1 has gone to (1,1,1)")

    
    go1.destroy_node()
    #goto.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()