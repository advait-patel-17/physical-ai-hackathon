#!/usr/bin/env python
"""
Deploy Cosmos Policy on OMX Arms using LeRobot

This script connects to a remote Cosmos Policy server and uses it to control
an OMX robot arm in real-time.

Prerequisites:
1. Start the Cosmos Policy server on a GPU machine:
   
   # On GPU server:
   cd cosmos-policy
   uv run -m cosmos_policy.experiments.robot.aloha.deploy \\
       --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_... \\
       --ckpt_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B \\
       --config_file cosmos_policy/config/config.py \\
       --host 0.0.0.0 \\
       --port 8777 \\
       ...

2. Connect your OMX arm and configure cameras

3. Run this script:
   
   python deploy_cosmos_omx.py \\
       --server_url http://your-gpu-server:8777 \\
       --robot_port /dev/ttyUSB0 \\
       --task "pick up the red block" \\
       --camera_index 0 \\
       --fps 10 \\
       --duration 60

Usage:
    python deploy_cosmos_omx.py --help

Example with two cameras:
    python deploy_cosmos_omx.py \\
        --server_url http://192.168.1.100:8777 \\
        --robot_port /dev/ttyUSB0 \\
        --task "pick up the cup and place it on the plate" \\
        --cameras '{"top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480}}' \\
        --third_person_camera top \\
        --fps 10 \\
        --duration 120
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

from cosmos_policy_client import CosmosPolicyClient, CosmosPolicyClientConfig

logger = logging.getLogger(__name__)


@dataclass
class DeployConfig:
    """Configuration for Cosmos Policy deployment on OMX arm."""
    
    # Cosmos Policy server
    server_url: str = "http://localhost:8777"
    server_timeout: float = 30.0
    
    # Robot configuration
    robot_port: str = "/dev/ttyUSB0"
    robot_type: str = "omx_follower"
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | None = 10.0  # Safety limit for position changes
    
    # Camera configuration (JSON string or dict)
    cameras: dict = field(default_factory=lambda: {
        "top": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
        }
    })
    
    # Camera mapping for Cosmos Policy
    third_person_camera: str = "top"
    left_wrist_camera: str | None = None
    right_wrist_camera: str | None = None
    
    # Task description
    task: str = "pick up the object"
    
    # Control parameters
    fps: float = 10.0  # Control loop frequency
    duration: float = 60.0  # Total runtime in seconds
    
    # Action chunking
    num_open_loop_steps: int = 5  # Steps before requesting new chunk
    
    # Display
    display_data: bool = False
    verbose: bool = False


class RobotController:
    """Thread-safe robot controller wrapper."""
    
    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()
    
    def get_observation(self) -> dict[str, Any]:
        with self.lock:
            return self.robot.get_observation()
    
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        with self.lock:
            return self.robot.send_action(action)
    
    @property
    def observation_features(self):
        return self.robot.observation_features
    
    @property
    def action_features(self):
        return self.robot.action_features


def control_loop(
    robot: RobotController,
    policy_client: CosmosPolicyClient,
    config: DeployConfig,
    shutdown_event: Event,
) -> None:
    """Main control loop that gets observations and sends actions."""
    
    logger.info(f"Starting control loop at {config.fps} Hz for {config.duration}s")
    logger.info(f"Task: {config.task}")
    
    step_interval = 1.0 / config.fps
    step_count = 0
    start_time = time.perf_counter()
    
    # Track timing statistics
    obs_times = []
    action_times = []
    send_times = []
    
    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time
            
            # Check duration limit
            if elapsed >= config.duration:
                logger.info("Duration limit reached")
                break
            
            # Get observation from robot
            obs_start = time.perf_counter()
            try:
                obs = robot.get_observation()
            except Exception as e:
                logger.error(f"Failed to get observation: {e}")
                continue
            obs_time = time.perf_counter() - obs_start
            obs_times.append(obs_time)
            
            # Request new action chunk periodically or when buffer is low
            force_new_chunk = (step_count % config.num_open_loop_steps == 0)
            
            # Get action from policy server
            action_start = time.perf_counter()
            try:
                action = policy_client.get_action(
                    observation=obs,
                    task=config.task,
                    force_new_chunk=force_new_chunk,
                )
            except Exception as e:
                logger.error(f"Failed to get action: {e}")
                continue
            action_time = time.perf_counter() - action_start
            action_times.append(action_time)
            
            # Send action to robot
            send_start = time.perf_counter()
            try:
                sent_action = robot.send_action(action)
            except Exception as e:
                logger.error(f"Failed to send action: {e}")
                continue
            send_time = time.perf_counter() - send_start
            send_times.append(send_time)
            
            step_count += 1
            
            # Display info
            if config.verbose and step_count % 10 == 0:
                avg_obs = np.mean(obs_times[-10:]) * 1000
                avg_action = np.mean(action_times[-10:]) * 1000
                avg_send = np.mean(send_times[-10:]) * 1000
                logger.info(
                    f"Step {step_count}: obs={avg_obs:.1f}ms, "
                    f"action={avg_action:.1f}ms, send={avg_send:.1f}ms"
                )
            
            # Wait for next step
            loop_time = time.perf_counter() - loop_start
            sleep_time = max(0, step_interval - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif config.verbose:
                logger.warning(f"Loop took {loop_time*1000:.1f}ms (target: {step_interval*1000:.1f}ms)")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Log statistics
        total_time = time.perf_counter() - start_time
        actual_fps = step_count / total_time if total_time > 0 else 0
        logger.info(f"Control loop finished: {step_count} steps in {total_time:.1f}s ({actual_fps:.1f} Hz)")
        
        if obs_times:
            logger.info(f"Observation time: avg={np.mean(obs_times)*1000:.1f}ms, max={np.max(obs_times)*1000:.1f}ms")
        if action_times:
            logger.info(f"Action time: avg={np.mean(action_times)*1000:.1f}ms, max={np.max(action_times)*1000:.1f}ms")
        if send_times:
            logger.info(f"Send time: avg={np.mean(send_times)*1000:.1f}ms, max={np.max(send_times)*1000:.1f}ms")


def make_robot(config: DeployConfig):
    """Create and connect to the robot."""
    from lerobot.cameras import CameraConfig
    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
    
    # Parse camera configs
    camera_configs = {}
    for name, cam_cfg in config.cameras.items():
        cam_type = cam_cfg.get("type", "opencv")
        
        if cam_type == "opencv":
            from lerobot.cameras.opencv import OpenCVCameraConfig
            camera_configs[name] = OpenCVCameraConfig(
                index_or_path=cam_cfg.get("index_or_path", 0),
                fps=cam_cfg.get("fps", 30),
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
            )
        elif cam_type == "realsense":
            from lerobot.cameras.realsense import RealSenseCameraConfig
            camera_configs[name] = RealSenseCameraConfig(
                serial_number=cam_cfg.get("serial_number"),
                fps=cam_cfg.get("fps", 30),
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
            )
        else:
            raise ValueError(f"Unknown camera type: {cam_type}")
    
    # Create robot config
    robot_config = OmxFollowerConfig(
        port=config.robot_port,
        cameras=camera_configs,
        disable_torque_on_disconnect=config.disable_torque_on_disconnect,
        max_relative_target=config.max_relative_target,
    )
    
    # Create and connect robot
    robot = OmxFollower(robot_config)
    robot.connect()
    
    return robot


def make_policy_client(config: DeployConfig) -> CosmosPolicyClient:
    """Create the Cosmos Policy client."""
    
    # Determine proprio keys based on robot type
    if config.robot_type == "omx_follower":
        proprio_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
    else:
        # Generic default
        proprio_keys = []
    
    client_config = CosmosPolicyClientConfig(
        server_url=config.server_url,
        timeout=config.server_timeout,
        third_person_camera=config.third_person_camera,
        left_wrist_camera=config.left_wrist_camera,
        right_wrist_camera=config.right_wrist_camera,
        proprio_keys=proprio_keys,
        num_open_loop_steps=config.num_open_loop_steps,
    )
    
    return CosmosPolicyClient(client_config)


def parse_args() -> DeployConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy Cosmos Policy on OMX arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Server
    parser.add_argument(
        "--server_url", type=str, default="http://localhost:8777",
        help="Cosmos Policy server URL"
    )
    parser.add_argument(
        "--server_timeout", type=float, default=30.0,
        help="Server request timeout in seconds"
    )
    
    # Robot
    parser.add_argument(
        "--robot_port", type=str, default="/dev/ttyUSB0",
        help="Serial port for robot"
    )
    parser.add_argument(
        "--robot_type", type=str, default="omx_follower",
        help="Robot type (omx_follower, so100_follower, etc.)"
    )
    parser.add_argument(
        "--max_relative_target", type=float, default=10.0,
        help="Safety limit for position changes (set to None to disable)"
    )
    
    # Cameras
    parser.add_argument(
        "--cameras", type=str, default=None,
        help='Camera configuration as JSON string, e.g., \'{"top": {"type": "opencv", "index_or_path": 0}}\''
    )
    parser.add_argument(
        "--camera_index", type=int, default=0,
        help="Default camera index (if --cameras not provided)"
    )
    parser.add_argument(
        "--third_person_camera", type=str, default="top",
        help="Camera name for third-person view"
    )
    parser.add_argument(
        "--left_wrist_camera", type=str, default=None,
        help="Camera name for left wrist view"
    )
    parser.add_argument(
        "--right_wrist_camera", type=str, default=None,
        help="Camera name for right wrist view"
    )
    
    # Task
    parser.add_argument(
        "--task", type=str, required=True,
        help="Natural language task description"
    )
    
    # Control
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Control loop frequency in Hz"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Total runtime in seconds"
    )
    parser.add_argument(
        "--num_open_loop_steps", type=int, default=5,
        help="Number of steps before requesting new action chunk"
    )
    
    # Display
    parser.add_argument(
        "--display_data", action="store_true",
        help="Display observation and action data"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Build camera config
    if args.cameras:
        cameras = json.loads(args.cameras)
    else:
        cameras = {
            "top": {
                "type": "opencv",
                "index_or_path": args.camera_index,
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        }
    
    # Handle max_relative_target
    max_rel = args.max_relative_target
    if max_rel is not None and max_rel <= 0:
        max_rel = None
    
    return DeployConfig(
        server_url=args.server_url,
        server_timeout=args.server_timeout,
        robot_port=args.robot_port,
        robot_type=args.robot_type,
        max_relative_target=max_rel,
        cameras=cameras,
        third_person_camera=args.third_person_camera,
        left_wrist_camera=args.left_wrist_camera,
        right_wrist_camera=args.right_wrist_camera,
        task=args.task,
        fps=args.fps,
        duration=args.duration,
        num_open_loop_steps=args.num_open_loop_steps,
        display_data=args.display_data,
        verbose=args.verbose,
    )


def main():
    """Main entry point."""
    
    # Parse arguments
    config = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("Cosmos Policy Deployment on OMX Arm")
    logger.info("=" * 60)
    logger.info(f"Server: {config.server_url}")
    logger.info(f"Robot port: {config.robot_port}")
    logger.info(f"Task: {config.task}")
    logger.info(f"FPS: {config.fps}, Duration: {config.duration}s")
    logger.info("=" * 60)
    
    # Setup shutdown event
    shutdown_event = Event()
    
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    robot = None
    policy_client = None
    
    try:
        # Create policy client
        logger.info("Creating Cosmos Policy client...")
        policy_client = make_policy_client(config)
        
        # Create and connect robot
        logger.info("Connecting to robot...")
        robot = make_robot(config)
        robot_controller = RobotController(robot)
        logger.info("Robot connected!")
        
        # Print robot features
        logger.info(f"Observation features: {list(robot.observation_features.keys())}")
        logger.info(f"Action features: {list(robot.action_features.keys())}")
        
        # Small delay to let robot settle
        time.sleep(0.5)
        
        # Run control loop
        control_loop(
            robot=robot_controller,
            policy_client=policy_client,
            config=config,
            shutdown_event=shutdown_event,
        )
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        if policy_client:
            policy_client.close()
        
        if robot:
            try:
                robot.disconnect()
                logger.info("Robot disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")
        
        logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
