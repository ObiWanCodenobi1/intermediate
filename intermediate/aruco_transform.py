#!usr/bin/env python 3
import rclpy
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Point
from icuas25_msgs.msg import TargetInfo
import tf2_ros

import numpy as np
from builtin_interfaces.msg import Duration

import time

class global_aruco(Node):
    def __init__(self):
        super().__init__("global_aruco")

        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer,self)
        
        self.no_of_drones = 5

        for i in range(1, self.no_of_drones+1):
            self.tf_buffer.wait_for_transform_async(f"cf_{i}","world",rclpy.time.Time())
            time.sleep(0.01)
        

        self.cf_poses = {
        f"cf_{i}": {
            "position": {"x": None, "y": None, "z": None},
            "orientation": {"x": None, "y": None, "z": None, "w": None}
        }
        for i in range(1, self.no_of_drones+1)
        }

        self.global_pose_subscriptions = []
        for i in range(1, self.no_of_drones+1):
            self.global_pose_subscriptions.append(self.create_subscription(ArucoMarkers, f"/cf_{i}/aruco_markers", lambda msg, drone_id=i: self.global_aruco_callback(msg, drone_id), 10))

        self.pub = self.create_publisher(TargetInfo, 'target_found', 10)
        self.detected = []
    
    

    def global_aruco_callback(self,msg:ArucoMarkers, drone_id):
        for i, marker_id in enumerate(msg.marker_ids):
            x_position = msg.poses[i].position.x
            y_position = msg.poses[i].position.y
            z_position = msg.poses[i].position.z
            x_orientation = msg.poses[i].orientation.x
            y_orientation = msg.poses[i].orientation.y
            z_orientation = msg.poses[i].orientation.z
            w_orientation = msg.poses[i].orientation.w

            
            self.get_logger().info(
                f"{drone_id}: Marker {marker_id} -> "
                f"Position (x: {x_position}, y: {y_position}, z: {z_position}), "
                f"Orientation (x: {x_orientation}, y: {y_orientation}, z: {z_orientation}, w: {w_orientation})"
            )
            

            try:
                transform = self.tf_buffer.lookup_transform(target_frame='world', source_frame=f"cf_{drone_id}", time=rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
            
            global_pos = self.transform_marker_to_global(msg.poses[i],transform)
            # self.get_logger().info(f"ArUco Marker Frame: {msg.header.frame_id}")
            # self.get_logger().info(f"Drone {drone_id} Local Marker Position: {marker_id.position}")
            # self.get_logger().info(f"Drone {drone_id} Transform: {transform}")

            self.get_logger().info(f"Aruco {marker_id} Global Position : {global_pos}")
            self.target_info = TargetInfo()
            self.target_info.id = marker_id
            self.target_info.location.x = global_pos[0]
            self.target_info.location.y = global_pos[1]
            self.target_info.location.z = global_pos[2]
            if marker_id not in self.detected:
                self.detected.append(marker_id)
                self.pub.publish(self.target_info)
            
    def normalize_quaternion(self, quaternion):
        norm = np.linalg.norm(quaternion)
        return [q / norm for q in quaternion]

    def Yaw(self,theta):
        return np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-1*np.sin(theta), np.cos(theta), 0],
            [0, 0 , 1]
        ])
    
    def Roll(self, theta):
        return np.array([
            [1, 0 ,0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -1*np.sin(theta), np.cos(theta)]
        ])
    
    def Pitch(self, theta):
        return np.array([
            [np.cos(theta),0,-1*np.sin(theta)],
            [0,1,0],
            [np.sin(theta),0, np.cos(theta)]
        ])
    
    
        
    def transform_marker_to_global(self, marker, transform):
        
        local_pos = np.array([marker.position.x, marker.position.y, marker.position.z], dtype=np.float64)

        rotation = transform.transform.rotation
        quaternion = self.normalize_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)

        translation = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z], dtype=np.float64)
        global_pos = (rotation_matrix @ self.Yaw(np.pi/2) @ self.Roll(np.pi/2)) @ local_pos + translation

        return global_pos
    
    def quaternion_to_rotation_matrix(self, quaternion):
        x, y, z, w = quaternion
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])


def main(args = None):
    rclpy.init(args=args)
    node = global_aruco()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__== "__main__":
    main()