"""
New script to convert raw data to LeRobotDataset format and push to hub
"""

import os
import sys

import numpy as np
import torch

import h5py
from pathlib import Path
import shutil
import argparse

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# joint limits for normalization
# BASE_VEL_HIGH = np.array([0.1, 0.15, 0.3])
# BASE_VEL_LOW = np.array([-0.1, -0.15, -0.3])
BASE_VEL_HIGH = np.array([1., 1., 1.]) # the state seems to be exceeding this limit. Using null limit for now.
BASE_VEL_LOW = np.array([-1., -1., -1.])
TORSO_JOINT_HIGH = np.array([1.8326, 2.5307, 1.8326, 3.0543])
TORSO_JOINT_LOW = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
LEFT_ARM_JOINT_HIGH = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
LEFT_ARM_JOINT_LOW = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
RIGHT_ARM_JOINT_HIGH = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
RIGHT_ARM_JOINT_LOW = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])


def create_r1_features_for_lerobot() -> dict:
    """
    Return a hardcoded dictionary of features for a simpler subset
    of the raw galaxea r1 HDF5 data: three cameras, a single 'robot state' vector,
    one 'action' vector, plus 'timestamp'.
    """
    # Example shapes: 3 cameras are stored as “video” with shape [3, 94, 168].
    # Robot state is a 1D vector (16).
    # Action is another 1D vector (18).
    # Timestamp is scalar (1).
    return {
        # Robot state
        "observation.state": {
            "dtype": "float32",
            "shape": (21,),  # base + torso + left arm + left gripper + right arm + right gripper
        },
        # Robot action
        "action": { # for lerobot implementation
        # "actions": { # for pi0 Jax implementation
            "dtype": "float32",
            "shape": (21,),
        }, 
        # Cameras as 'video' => will become MP4 if you run ds.consolidate(...).
        "observation.images.head": {
            "dtype": "video",
            "shape": (3, 94, 168),  # Keep original [C, H, W] shape
            "names": ["channels", "height", "width"],  # Adjust names to match shape order
            "info": {
                "video.fps": 30.0,
                "video.height": 94,
                "video.width": 168,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.left_wrist": {
            "dtype": "video",
            "shape": (3, 94, 168),
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": 30.0,
                "video.height": 94,
                "video.width": 168,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": (3, 94, 168),
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": 30.0,
                "video.height": 94,
                "video.width": 168,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.pointcloud.xyz": {
            "dtype": "float32",
            "shape": (4096, 3), # all pointclouds have already been downsampled to 4096
        },
        "observation.pointcloud.rgb": {
            "dtype": "uint8",
            "shape": (4096, 3), # all pointclouds have already been downsampled to 4096
        },
        "observation.pointcloud.padding_mask": {
            "dtype": "bool",
            "shape": (4096,), # all pointclouds have already been downsampled to 4096
        }
        # You could add more if needed...
    }

def clip_joints(torso, left_arm, right_arm):
    """
    Clip torso, left_arm, right_arm to be within the joint limits
    """
    torso = np.clip(torso, TORSO_JOINT_LOW, TORSO_JOINT_HIGH)
    left_arm = np.clip(left_arm, LEFT_ARM_JOINT_LOW, LEFT_ARM_JOINT_HIGH)
    right_arm = np.clip(right_arm, RIGHT_ARM_JOINT_LOW, RIGHT_ARM_JOINT_HIGH)
    return torso, left_arm, right_arm

def normalize_joints(torso, left_arm, right_arm):
    """
    Normalize torso, arms action or state to [-1, 1]
    input:
        torso: (4,) or (N, 4)
        left_arm: (6,) or (N, 6)
        right_arm: (6,) or (N, 6)
    output:
        torso: (4,) or (N, 4)
        left_arm: (6,) or (N, 6)
        right_arm: (6,) or (N, 6)
    """
    torso = ((torso - TORSO_JOINT_LOW) / (TORSO_JOINT_HIGH - TORSO_JOINT_LOW)) * 2 - 1  
    left_arm = ((left_arm - LEFT_ARM_JOINT_LOW) / (LEFT_ARM_JOINT_HIGH - LEFT_ARM_JOINT_LOW)) * 2 - 1
    right_arm = ((right_arm - RIGHT_ARM_JOINT_LOW) / (RIGHT_ARM_JOINT_HIGH - RIGHT_ARM_JOINT_LOW)) * 2 - 1
    return torso, left_arm, right_arm


def add_episode_from_hdf5(
    ds: LeRobotDataset,
    ep_group: h5py.Group,
    ep_idx: int,
    task_name: str,
    fps: float = 30.0,
    normalize: bool = False,
):
    """
    Reads a single 'demo_X' group from the HDF5 file, extracts per-frame data, and
    adds them to the LeRobotDataset 'ds' one frame at a time. 
    Finally calls ds.save_episode(...) to finalize the episode.

    Args:
        ds: The LeRobotDataset to which we add frames.
        ep_group: The h5py group, e.g. file['demo_0'], containing the episode's raw data.
        ep_idx: Episode index (for logging or debugging).
        task: A string describing this task (will be saved in ds.save_episode(task=...)).
        fps: Frames per second used for timestamps. 
        normalize: Whether to normalize the joints to [-1, 1]
    """
    print(f"Converting episode {ep_idx} with {fps} fps. Group keys: {list(ep_group.keys())}")

    # 1) Determine the number of frames in this episode
    #    e.g., from 'action/left_arm' or any consistent dataset
    num_frames = ep_group['action/left_arm'].shape[0]
    print(f"Episode {ep_idx}: {num_frames} frames.")

    # 2) Pre-load arrays from the HDF5 so we only do I/O once
    #    (For large data, you might want to do it lazily, but for clarity we show it here.)
    left_arm_action = ep_group['action/left_arm'][:]
    right_arm_action = ep_group['action/right_arm'][:]
    mobile_base_action = ep_group['action/mobile_base'][:]
    torso_action = ep_group['action/torso'][:]
    left_gripper_action = ep_group['action/left_gripper'][:] / 100.0 # normalize to 0-1
    right_gripper_action = ep_group['action/right_gripper'][:] / 100.0
    # state data
    base_vel = ep_group['obs/odom/linear_velocity'][:]
    torso_pos = ep_group['obs/joint_state/torso/joint_position'][:]
    left_pos = ep_group['obs/joint_state/left_arm/joint_position'][:,:-1] # discard last dim
    left_gripper_pos = ep_group['obs/gripper_state/left_gripper/gripper_position'][:] / 100.0 # normalize to 0-1
    right_pos = ep_group['obs/joint_state/right_arm/joint_position'][:,:-1] # discard last dim
    right_gripper_pos = ep_group['obs/gripper_state/right_gripper/gripper_position'][:] / 100.0 # normalize to 0-1

    # clip invalid values
    base_vel = np.clip(base_vel, -1, 1)
    torso_pos, left_pos, right_pos = clip_joints(torso_pos, left_pos, right_pos)
    torso_action, left_arm_action, right_arm_action = clip_joints(torso_action, left_arm_action, right_arm_action)

    if normalize:
        torso_action, left_arm_action, right_arm_action = normalize_joints(torso_action, left_arm_action, right_arm_action)
        torso_pos, left_pos, right_pos = normalize_joints(torso_pos, left_pos, right_pos)

    # camera images
    # shape might be (num_frames, H, W, 3). We'll transpose to (3, H, W) per frame
    head_imgs = ep_group['obs/rgb/head/img'][:]         # shape (N,94,168,3)
    left_wrist_imgs = ep_group['obs/rgb/left_wrist/img'][:]
    right_wrist_imgs = ep_group['obs/rgb/right_wrist/img'][:]

    pc_xyz = ep_group['obs/point_cloud/fused/xyz'][:]
    pc_rgb = ep_group['obs/point_cloud/fused/rgb'][:]
    pc_padding_mask = ep_group['obs/point_cloud/fused/padding_mask'][:]

    # 3) For each frame, build a dictionary & call ds.add_frame(...)
    for t in range(num_frames):
        frame_data = {}

        # 3.1) Cameras
        # Transpose each image from (H,W,3) -> (3,H,W)
        frame_data["observation.images.head"] = np.transpose(head_imgs[t], (2,0,1))
        frame_data["observation.images.left_wrist"] = np.transpose(left_wrist_imgs[t], (2,0,1))
        frame_data["observation.images.right_wrist"] = np.transpose(right_wrist_imgs[t], (2,0,1))

        frame_data["observation.pointcloud.xyz"] = pc_xyz[t]
        frame_data["observation.pointcloud.rgb"] = pc_rgb[t]
        frame_data["observation.pointcloud.padding_mask"] = pc_padding_mask[t]

        # 3.2) State vector
        # e.g. concat left_pos[t], right_pos[t], torso_pos[t]
        #   left_pos[t] shape => (6,) or (7,)
        #   right_pos[t] shape => (6,) or (7,)
        #   torso_pos[t] shape => (4,) ...
        # Adjust as needed
        # state_vec = np.concatenate([left_pos[t], right_pos[t], torso_pos[t]], axis=0)
        state_vec = np.concatenate([
            base_vel[t], #(3,)
            torso_pos[t], #(4,)
            left_pos[t], #(6,) discard last dim
            left_gripper_pos[t:t+1], #(1,)
            right_pos[t], #(6,)
            right_gripper_pos[t:t+1], #(1,)
        ], axis=0)
        frame_data["observation.state"] = np.array(state_vec, dtype=np.float32).flatten()

        # 3.3) Action vector
        act_vec = np.concatenate([
            mobile_base_action[t], #(3,)
            torso_action[t], #(4,)
            left_arm_action[t], #(6,)
            left_gripper_action[t:t+1], #(1,)
            right_arm_action[t], #(6,)
            right_gripper_action[t:t+1], #(1,)
        ], axis=0)
        frame_data["action"] = np.array(act_vec, dtype=np.float32).flatten() # for lerobot implementation
        # frame_data["actions"] = np.array(act_vec, dtype=np.float32) # for lerobot implementation

        # 3.4) Timestamp
        frame_data["timestamp"] = np.array(float(t) / fps, dtype=np.float32)[np.newaxis]

        frame_data["task"] = task_name

        # 3.5) Add the frame
        ds.add_frame(frame_data)

    # 4) Finalize the episode
    ds.save_episode()
    #    (If you want immediate .mp4 encoding, do encode_videos=True instead)
    #    Typically, you'd do ds.consolidate(...) once at the very end for all episodes

    print(f"Episode {ep_idx} from group {ep_group.name} added to ds.")

def convert_single_hdf5_to_lerobot(raw_path: Path, task_name: str, ds: LeRobotDataset, fps: int = 30, normalize: bool = False):
    """
    Finds a .hdf5 file in raw_dir, derives task from its filename, 
    iterates over each 'demo_*' group (episode) in the file, 
    and adds those episodes to the given LeRobotDataset.
    
    Args:
        raw_dir: Directory containing your .hdf5 file(s).
        ds: A LeRobotDataset object that was already created 
            via LeRobotDataset.create(...).
        fps: Frame rate to associate with each episode.
    """
    hdf5_path = raw_path
    # task_name = hdf5_path.stem  # e.g. "task_name" from "task_name.hdf5"
    print(f"Processing task: {task_name} from {hdf5_path}")

    # 3. Open the file
    with h5py.File(hdf5_path, "r") as hf:
        # Suppose your episodes are stored as "demo_0", "demo_1", ...
        # If that's consistent, gather them:
        episode_groups = [key for key in hf.keys() if key.startswith("demo_")]
        # sort episode_groups by the number in the string
        episode_groups.sort(key=lambda x: int(x.split("_")[1]))
        print(f"Found episodes: {episode_groups}")

        for i, ep_key in enumerate(episode_groups):
            if i == 0:
                continue # skip episode 0 -- have issue with data

            ep_group = hf[ep_key]
            print(f"Adding episode {ep_key} => index {i} to dataset.")
            add_episode_from_hdf5(
                ds=ds,
                ep_group=ep_group,
                ep_idx=i,
                task_name=task_name,
                fps=fps,
                normalize=normalize,
            )

            # break
            # only add 1 episode for now
            # if i > 9:
            # if i > 0:
            #     break

    print("Done adding all episodes. You can now ds.consolidate() and ds.push_to_hub() if you like.")

def main(repo_id: str, 
         raw_path: str,
         task_name: str,
         normalize: bool = False):
    features = create_r1_features_for_lerobot()

    # repo_id = "s-tian/galaxea-r1-shelf-full-normalized_pc"
    # repo_id = "Yiheyihe/galaxea-r1-shelf-full-normalized"
    # repo_id = "Yiheyihe/galaxea-r1-shelf-debug-normalized"
    # repo_id = "Yiheyihe/galaxea-r1-shelf-debug-pi0Jax"
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=None,
        features=features,
        use_videos=True,
    )
    # ds = LeRobotDataset(
    #     repo_id=repo_id,
    #     root="/viscam/u/stian/lerobot/s-tian/galaxea-r1-shelf-full-normalized",
    # )

    raw_path = Path(raw_path)

    # dataset_path = Path(os.path.join("/svl/u/stian/lerobot/.hf_cache/lerobot/", repo_id))
    # # remove the dataset_path if it exists
    # if dataset_path.exists():
    #     print(f"Removing existing dataset at {dataset_path}")
    #     shutil.rmtree(dataset_path)

    convert_single_hdf5_to_lerobot(raw_path, task_name, ds, normalize=normalize)

    # from IPython import embed; embed(); exit(0)
    # ds.consolidate()
    ds.push_to_hub()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="s-tian/galaxea-r1-shelf-full-normalized_pc", 
                        help="huggingface repo name to upload to")
    parser.add_argument("--hdf5_path", type=str, default="/viscam/projects/3dvla/", 
                        help="Path to the directory containing HDF5 files")
    parser.add_argument("--task_name", type=str, default="shelf", 
                        help="Task name to use for the dataset")

    args = parser.parse_args()
    main(repo_id=args.repo_id, 
         raw_path=args.hdf5_path, 
         task_name=args.task_name, 
         normalize=True)