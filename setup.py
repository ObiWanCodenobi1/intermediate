from setuptools import find_packages, setup

package_name = 'intermediate'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'solution = intermediate.intermediate_solution:main',
            'go = intermediate.goto:main',
            'transform = intermediate.aruco_transform:main',
            'map_publisher = intermediate.map_publisher:main',
            'exploration_map = intermediate.exploration_map:main',
            'drone_trajectory_controller = intermediate.drone_trajectory_controller:main',
            'takeoff_all = intermediate.takeoff_all:main',
            'frontier_detection = intermediate.frontier_detection:main',
            'swarm_controller: intermediate.swarm_controller:main'
        ],
    },
)
