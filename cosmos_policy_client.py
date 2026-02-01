#!/usr/bin/env python
"""
Cosmos Policy Client for LeRobot

This module provides a client to connect to the NVIDIA Cosmos Policy server
and use it with LeRobot-compatible robots (e.g., OMX arms).

The Cosmos Policy server exposes an /act endpoint that accepts observations
and returns action chunks. This client handles the communication, image
formatting, and action processing.

Reference: https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/experiments/robot/aloha/deploy.py

Usage:
    # Start the Cosmos Policy server (on GPU machine):
    uv run -m cosmos_policy.experiments.robot.aloha.deploy \\
        --config cosmos_predict2_2b_480p_aloha_... \\
        --ckpt_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B \\
        ...

    # Then run this client with OMX arm:
    python cosmos_policy_client.py \\
        --server_url http://your-server:8777 \\
        --task "pick up the cup" \\
        --robot_port /dev/ttyUSB0
"""

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class CosmosPolicyClientConfig:
    """Configuration for the Cosmos Policy client."""
    
    # Server connection
    server_url: str = "http://localhost:8777"
    timeout: float = 30.0  # Request timeout in seconds
    
    # Image configuration
    # Cosmos Policy expects specific image sizes depending on training
    image_width: int = 640
    image_height: int = 480
    use_jpeg_compression: bool = True  # Compress images for faster transfer
    jpeg_quality: int = 85
    
    # Camera mapping from LeRobot camera names to Cosmos observation keys
    # Adjust these based on your camera setup
    third_person_camera: str | None = "top"  # Camera name for third-person view
    left_wrist_camera: str | None = None  # Camera name for left wrist (if applicable)
    right_wrist_camera: str | None = None  # Camera name for right wrist (if applicable)
    
    # Proprioception configuration
    # Motor names to include in proprio (in order)
    proprio_keys: list[str] = field(default_factory=lambda: [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ])
    
    # Action configuration
    # Number of actions to return from each chunk
    num_open_loop_steps: int = 1  # Execute this many steps before requesting new chunk
    action_chunk_size: int = 50  # Size of action chunk from server
    
    # Whether to double encode JSON (some servers require this)
    double_encode: bool = False


class CosmosPolicyClient:
    """
    Client for NVIDIA Cosmos Policy server.
    
    This client connects to a Cosmos Policy server and provides an interface
    compatible with LeRobot's policy inference pipeline.
    
    Example:
        client = CosmosPolicyClient(CosmosPolicyClientConfig(
            server_url="http://gpu-server:8777",
            third_person_camera="top",
        ))
        
        # Get action from observation
        obs = robot.get_observation()
        action = client.get_action(obs, task="pick up the red block")
        robot.send_action(action)
    """
    
    def __init__(self, config: CosmosPolicyClientConfig):
        self.config = config
        self.session = requests.Session()
        self._action_buffer: list[np.ndarray] = []
        self._buffer_index: int = 0
        self._last_task: str = ""
        
        # Verify server is reachable
        self._check_server_connection()
    
    def _check_server_connection(self) -> None:
        """Check if the server is reachable."""
        try:
            # Try a simple request to verify connectivity
            response = self.session.get(
                self.config.server_url.rstrip('/'),
                timeout=5.0
            )
            logger.info(f"Connected to Cosmos Policy server at {self.config.server_url}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Could not connect to server at {self.config.server_url}. "
                "Make sure the Cosmos Policy server is running."
            )
        except Exception as e:
            logger.warning(f"Server check failed: {e}")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string, optionally with JPEG compression."""
        if image is None:
            return None
        
        # Ensure image is in correct format (H, W, C) and uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Resize if needed
        if image.shape[:2] != (self.config.image_height, self.config.image_width):
            image = cv2.resize(image, (self.config.image_width, self.config.image_height))
        
        if self.config.use_jpeg_compression:
            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            _, buffer = cv2.imencode('.jpg', image, encode_params)
            return base64.b64encode(buffer).decode('utf-8')
        else:
            # Encode as PNG (lossless but larger)
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')
    
    def _prepare_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Convert LeRobot observation format to Cosmos Policy format.
        
        LeRobot obs format:
            - Images: numpy arrays (H, W, C) or (C, H, W)
            - Motors: {motor_name}.pos floats
            
        Cosmos Policy format:
            - third_person_image: base64 encoded image
            - left_wrist_image: base64 encoded image (optional)
            - right_wrist_image: base64 encoded image (optional)
            - proprio: list of floats
        """
        cosmos_obs = {}
        
        # Process third-person image
        if self.config.third_person_camera and self.config.third_person_camera in obs:
            img = obs[self.config.third_person_camera]
            # Handle channel-first format (C, H, W) -> (H, W, C)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            cosmos_obs["third_person_image"] = self._format_image_for_server(img)
        
        # Process left wrist image
        if self.config.left_wrist_camera and self.config.left_wrist_camera in obs:
            img = obs[self.config.left_wrist_camera]
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            cosmos_obs["left_wrist_image"] = self._format_image_for_server(img)
        
        # Process right wrist image
        if self.config.right_wrist_camera and self.config.right_wrist_camera in obs:
            img = obs[self.config.right_wrist_camera]
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            cosmos_obs["right_wrist_image"] = self._format_image_for_server(img)
        
        # Process proprioception (joint positions)
        proprio = []
        for key in self.config.proprio_keys:
            if key in obs:
                proprio.append(float(obs[key]))
            else:
                # Try without .pos suffix
                key_base = key.replace(".pos", "")
                if key_base in obs:
                    proprio.append(float(obs[key_base]))
                else:
                    logger.warning(f"Proprio key {key} not found in observation")
                    proprio.append(0.0)
        
        cosmos_obs["proprio"] = proprio
        
        return cosmos_obs
    
    def _format_image_for_server(self, image: np.ndarray) -> np.ndarray:
        """Format image for server transmission."""
        if image is None:
            return None
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Resize if needed
        target_h, target_w = self.config.image_height, self.config.image_width
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h))
        
        return image.tolist()  # Convert to list for JSON serialization
    
    def _request_action_chunk(self, obs: dict[str, Any], task: str) -> np.ndarray:
        """Send observation to server and get action chunk."""
        
        # Prepare the request payload
        payload = {
            "observation": obs,
            "task_description": task,
        }
        
        try:
            # Import json_numpy for proper numpy array serialization
            try:
                import json_numpy
                json_numpy.patch()
                import json
                data = json.dumps(payload)
            except ImportError:
                import json
                data = json.dumps(payload, cls=NumpyEncoder)
            
            # Make the request
            response = self.session.post(
                f"{self.config.server_url.rstrip('/')}/act",
                data=data,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
            )
            
            response.raise_for_status()
            
            # Parse response
            if self.config.double_encode:
                import json
                result = json.loads(json.loads(response.text))
            else:
                result = response.json()
            
            if isinstance(result, str) and result == "error":
                raise RuntimeError("Server returned error")
            
            actions = np.array(result["actions"])
            
            # Log additional info if available
            if "value_prediction" in result:
                logger.debug(f"Value prediction: {result['value_prediction']:.4f}")
            
            return actions
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.config.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get_action(
        self,
        observation: dict[str, Any],
        task: str,
        force_new_chunk: bool = False,
    ) -> dict[str, float]:
        """
        Get the next action for the robot.
        
        This method manages action chunking: it requests a new action chunk
        from the server when the buffer is empty or when force_new_chunk is True,
        and returns actions from the buffer otherwise.
        
        Args:
            observation: Robot observation dict from robot.get_observation()
            task: Natural language task description
            force_new_chunk: If True, request new chunk even if buffer has actions
            
        Returns:
            Dictionary mapping motor names to position values
        """
        # Check if we need a new action chunk
        need_new_chunk = (
            force_new_chunk or
            len(self._action_buffer) == 0 or
            self._buffer_index >= len(self._action_buffer) or
            task != self._last_task
        )
        
        if need_new_chunk:
            logger.debug("Requesting new action chunk from server...")
            start_time = time.perf_counter()
            
            # Prepare observation in Cosmos format
            cosmos_obs = self._prepare_observation(observation)
            
            # Request action chunk from server
            action_chunk = self._request_action_chunk(cosmos_obs, task)
            
            # Update buffer
            self._action_buffer = list(action_chunk)
            self._buffer_index = 0
            self._last_task = task
            
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Got action chunk with {len(self._action_buffer)} actions in {elapsed:.3f}s")
        
        # Get next action from buffer
        if self._buffer_index < len(self._action_buffer):
            action_array = self._action_buffer[self._buffer_index]
            self._buffer_index += 1
        else:
            logger.warning("Action buffer exhausted, using last action")
            action_array = self._action_buffer[-1] if self._action_buffer else np.zeros(len(self.config.proprio_keys))
        
        # Convert action array to dict with motor names
        # Remove .pos suffix from proprio keys to get motor names
        motor_names = [k.replace(".pos", "") for k in self.config.proprio_keys]
        action_dict = {
            f"{name}.pos": float(action_array[i])
            for i, name in enumerate(motor_names)
            if i < len(action_array)
        }
        
        return action_dict
    
    def reset(self) -> None:
        """Reset the action buffer (e.g., at episode start)."""
        self._action_buffer = []
        self._buffer_index = 0
        self._last_task = ""
    
    def close(self) -> None:
        """Clean up resources."""
        self.session.close()


class NumpyEncoder(object):
    """JSON encoder for numpy arrays (fallback if json_numpy not available)."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# Make encoder work with json.dumps
import json
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)
    
    config = CosmosPolicyClientConfig(
        server_url="http://localhost:8777",
    )
    client = CosmosPolicyClient(config)
    
    # Create dummy observation
    dummy_obs = {
        "top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
    
    print("Testing Cosmos Policy Client...")
    print(f"Server URL: {config.server_url}")
    
    try:
        action = client.get_action(dummy_obs, task="pick up the cup")
        print(f"Received action: {action}")
    except Exception as e:
        print(f"Error (expected if server not running): {e}")
