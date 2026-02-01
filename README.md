# Cosmos Policy Client for LeRobot

Deploy NVIDIA Cosmos Policy on OMX robot arms using LeRobot.

## Overview

This project provides a client to connect to a remote [Cosmos Policy](https://github.com/NVlabs/cosmos-policy) server and use it to control robot arms (like OMX, SO100, Koch) through LeRobot's hardware abstraction.

### Architecture

```
┌─────────────────────┐          HTTP/JSON           ┌─────────────────────┐
│                     │ ◄─────────────────────────► │                     │
│   Robot Machine     │     POST /act               │   GPU Server        │
│   (Local)           │     {observation, task}     │   (Remote)          │
│                     │                             │                     │
│  ┌───────────────┐  │                             │  ┌───────────────┐  │
│  │ deploy_cosmos │  │                             │  │ Cosmos Policy │  │
│  │ _omx.py       │  │     ◄── {actions} ──        │  │ Server        │  │
│  └───────────────┘  │                             │  └───────────────┘  │
│         │           │                             │                     │
│         ▼           │                             │                     │
│  ┌───────────────┐  │                             │                     │
│  │ LeRobot       │  │                             │                     │
│  │ OMX Follower  │  │                             │                     │
│  └───────────────┘  │                             │                     │
│         │           │                             │                     │
│         ▼           │                             │                     │
│  ┌───────────────┐  │                             │                     │
│  │ OMX Robot Arm │  │                             │                     │
│  └───────────────┘  │                             │                     │
└─────────────────────┘                             └─────────────────────┘
```

## Quick Start

### 1. Start the Cosmos Policy Server (GPU Machine)

On a machine with a GPU, start the Cosmos Policy server:

```bash
# Clone cosmos-policy if you haven't
git clone https://github.com/NVlabs/cosmos-policy
cd cosmos-policy

# Run the policy server
uv run -m cosmos_policy.experiments.robot.aloha.deploy \
    --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only \
    --ckpt_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_third_person_image True \
    --use_wrist_image False \
    --num_wrist_images 0 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_t5_embeddings.pkl \
    --chunk_size 50 \
    --num_open_loop_steps 50 \
    --host 0.0.0.0 \
    --port 8777
```

The server will listen on `http://<gpu-machine-ip>:8777`.

### 2. Run the Client (Robot Machine)

On the machine connected to your robot:

```bash
cd physical-ai-hackathon

# Install dependencies
pip install requests numpy opencv-python

# Run the deployment script
python deploy_cosmos_omx.py \
    --server_url http://<gpu-server-ip>:8777 \
    --robot_port /dev/ttyUSB0 \
    --task "pick up the red block" \
    --camera_index 0 \
    --fps 10 \
    --duration 60
```

## Files

- **`cosmos_policy_client.py`** - HTTP client that communicates with the Cosmos Policy server
- **`deploy_cosmos_omx.py`** - Main deployment script for OMX arms

## Configuration Options

### Server Connection

| Option | Default | Description |
|--------|---------|-------------|
| `--server_url` | `http://localhost:8777` | Cosmos Policy server URL |
| `--server_timeout` | `30.0` | Request timeout in seconds |

### Robot

| Option | Default | Description |
|--------|---------|-------------|
| `--robot_port` | `/dev/ttyUSB0` | Serial port for the robot |
| `--robot_type` | `omx_follower` | Robot type |
| `--max_relative_target` | `10.0` | Safety limit for position changes |

### Cameras

| Option | Default | Description |
|--------|---------|-------------|
| `--cameras` | - | JSON camera config |
| `--camera_index` | `0` | Default camera index |
| `--third_person_camera` | `top` | Camera name for third-person view |
| `--left_wrist_camera` | `None` | Camera name for left wrist |
| `--right_wrist_camera` | `None` | Camera name for right wrist |

### Control

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | (required) | Natural language task description |
| `--fps` | `10.0` | Control loop frequency in Hz |
| `--duration` | `60.0` | Total runtime in seconds |
| `--num_open_loop_steps` | `5` | Steps before requesting new action chunk |

## Advanced Usage

### Multiple Cameras

```bash
python deploy_cosmos_omx.py \
    --server_url http://192.168.1.100:8777 \
    --robot_port /dev/ttyUSB0 \
    --task "pick up the cup" \
    --cameras '{
        "top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480},
        "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480}
    }' \
    --third_person_camera top \
    --left_wrist_camera wrist \
    --fps 10
```

### Using RealSense Camera

```bash
python deploy_cosmos_omx.py \
    --server_url http://192.168.1.100:8777 \
    --robot_port /dev/ttyUSB0 \
    --task "pick up the object" \
    --cameras '{
        "top": {
            "type": "realsense",
            "serial_number": "123456789",
            "width": 640,
            "height": 480,
            "fps": 30
        }
    }' \
    --third_person_camera top
```

### Using as a Library

```python
from cosmos_policy_client import CosmosPolicyClient, CosmosPolicyClientConfig

# Create client
config = CosmosPolicyClientConfig(
    server_url="http://gpu-server:8777",
    third_person_camera="top",
    proprio_keys=[
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ],
)
client = CosmosPolicyClient(config)

# Use with your own robot loop
while running:
    obs = robot.get_observation()
    action = client.get_action(obs, task="pick up the cup")
    robot.send_action(action)

client.close()
```

## Troubleshooting

### Connection Refused
Make sure the Cosmos Policy server is running and accessible:
```bash
curl http://<server-ip>:8777
```

### Slow Inference
- Increase `--num_open_loop_steps` to reduce server requests
- Ensure GPU is being used on the server
- Check network latency between client and server

### Robot Not Moving
1. Check serial port permissions: `sudo chmod 666 /dev/ttyUSB0`
2. Verify robot is calibrated
3. Check `--max_relative_target` isn't too restrictive

### Camera Issues
1. List available cameras: `python -c "from lerobot.cameras.opencv import OpenCVCamera; print(OpenCVCamera.find_cameras())"`
2. Check camera permissions
3. Verify camera index/path is correct

## Dependencies

- Python 3.10+
- LeRobot (included in this repo)
- requests
- numpy
- opencv-python

Optional:
- json_numpy (for better numpy serialization)

## References

- [Cosmos Policy](https://github.com/NVlabs/cosmos-policy) - NVIDIA's policy model
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace's robot learning framework
- [OMX](https://github.com/ROBOTIS-GIT/open_manipulator) - ROBOTIS Open Manipulator
