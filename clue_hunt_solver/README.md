# Clue Hunt Solver

ROS 2 Humble solution for the Clue Chain Hunt leader and follower robots.

## Nodes

- `hunt_node`: detects and authenticates ArUco/QR clue boards, estimates board poses in `map`, observes coloured pillars, interprets all clue commands, and sends Nav2 goals.
- `follower_node`: tracks rear marker 49 using only the follower camera/odometry and follows a trail of visual tag observations to avoid cutting corners. It does not subscribe to any leader topic or frame.

Neither node reads Gazebo ground truth or hard-codes practice board, pillar, or treasure positions.

## Build

```bash
cd ~/hunt_ws
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install -y python3-pyzbar libzbar0
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select clue_hunt_solver
source install/setup.bash
```

## Create the map once

The solution localizes every visual observation in `map`, so Nav2 needs a saved
2-D occupancy map.  Start mapping with Gazebo's rendering disabled (important in
VirtualBox) and RViz enabled:

```bash
mkdir -p ~/hunt_ws/maps
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch clue_hunt_navigation mapping.launch.py \
  gui:=false rviz:=true
```

Open another terminal and drive the leader around the complete inside perimeter:

```bash
source /opt/ros/humble/setup.bash
source ~/hunt_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

When the RViz map is complete, stop the robot and save it from a third terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/hunt_ws/install/setup.bash
ros2 run nav2_map_server map_saver_cli -f ~/hunt_ws/maps/arena
```

This creates `arena.yaml` and `arena.pgm`. Stop mapping with `Ctrl+C` only after
the save command reports success.

For submission, copy both saved files into this package's `maps/` directory:

```bash
cp ~/hunt_ws/maps/arena.yaml ~/hunt_ws/src/clue_hunt_solver/maps/
cp ~/hunt_ws/maps/arena.pgm  ~/hunt_ws/src/clue_hunt_solver/maps/
```

## Run the autonomous solution

Start the simulator and Nav2 with the saved map:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch clue_hunt_navigation navigation.launch.py \
  map:=$HOME/hunt_ws/src/clue-chain-hunt/clue_hunt_solver/maps/arena.yaml gui:=false rviz:=false
```

In another terminal:

```bash
source ~/hunt_ws/install/setup.bash
ros2 launch clue_hunt_solver hunt.launch.py
```

After a map exists, simulation + Nav2 + both solver nodes can instead be started with one command:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch clue_hunt_solver autonomy.launch.py \
  gui:=false rviz:=false
```

Watch the required outputs with:

```bash
ros2 topic echo /hunt/clues
ros2 topic echo /hunt/boards
ros2 topic echo /hunt/treasure
ros2 topic echo /leader/status
```

RViz bonus/debug topics are also provided:

```bash
ros2 topic echo /hunt/markers     # accepted boards, perceived pillars, treasure
ros2 topic echo /follower/path    # path in follower/odom (follower data only)
ros2 topic echo /follower/range   # measured distance to tag 49 in metres
```

The follower's path must be viewed with RViz fixed frame `follower/odom`. The
range topic makes the required 0.6--2.0 m separation directly measurable.

The package includes a saved practice map and uses it by default in
`autonomy.launch.py`. The image tests also include local board fixtures, so they
remain reproducible when this package is reviewed outside the organizer tree.

For a low-frame-rate VM, leave `confirmation_frames` at 2. On native hardware it can be raised to 3. Controller gains are in `config/solver_params.yaml`.

If VirtualBox previously showed blank blue camera frames or froze in Gazebo, use
`VMSVGA`, 128 MB video memory, disable 3-D acceleration, and retain
`LIBGL_ALWAYS_SOFTWARE=1`.  A renderer reported as `llvmpipe` is expected for
this VM configuration.
