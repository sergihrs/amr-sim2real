colcon build --symlink-install

. install/setup.bash

ros2 launch turtlebot3_bringup robot.launch.py
ros2 run amr_control teleoperation_node
ros2 run amr_control keyboard_node
ros2 launch launch/project_sim.launch.py
