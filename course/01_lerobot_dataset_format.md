# Chapter 1: The LeRobotDataset Format

## The Problem

Robot learning requires data — demonstrations of a task performed correctly. But robotics data is messy: it's multimodal (images, joint positions, forces), temporal (ordered sequences at specific frame rates), and heterogeneous (different robots have different numbers of joints, cameras, and sensors). Every research lab historically invented their own format, making it impossible to share data or reproduce results.

LeRobot solves this with a standardised dataset format: LeRobotDataset. Understanding this format deeply is the single most important skill for working with LeRobot — when your SO-101 arrives, 80% of your debugging will be data problems.

## The Format: Parquet + MP4

A LeRobotDataset has two storage types:

**Parquet files** store structured numerical data — joint positions, actions, timestamps, episode boundaries. One file per episode, stored at `data/chunk-000/episode_000000.parquet`. Parquet is a columnar binary format from the Apache ecosystem. It's much faster to read than CSV and supports efficient compression.

**MP4 videos** store camera observations. One video per episode per camera, stored at `videos/chunk-000/observation.images.top/episode_000000.mp4`. Videos use AV1 codec for high compression. A single 480×640 RGB frame is ~900KB raw. At 50fps over 400 frames, that's 360MB per episode. MP4 with AV1 brings this down to a few MB by exploiting temporal redundancy — consecutive frames look almost identical, so the encoder only stores the differences.

**info.json** describes the data contract: what features exist, their dtypes, shapes, and for motor data, which dimension maps to which joint.

```
dataset_repo/
├── info.json                           # Metadata: features, shapes, fps
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet      # Joint states, actions, timestamps
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.top/
            ├── episode_000000.mp4      # Camera feed for episode 0
            ├── episode_000001.mp4
            └── ...
```

## What's Inside Each Frame

When you index into a LeRobotDataset (`dataset[i]`), you get a Python dict with these keys:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `observation.state` | [14] | float32 | Current joint positions (radians) |
| `action` | [14] | float32 | Target joint positions |
| `observation.images.top` | [3, 480, 640] | float32 | Camera image (CHW, normalised to [0,1]) |
| `episode_index` | scalar | int64 | Which episode this frame belongs to |
| `frame_index` | scalar | int64 | Frame number within the episode |
| `timestamp` | scalar | float32 | Seconds since episode start |
| `task` | — | string | Task description text |
| `task_index` | scalar | int64 | Task ID |
| `next.done` | scalar | bool | True at the last frame of an episode |
| `index` | scalar | int64 | Global index in the flat dataset |

## The 14-Dimensional Action Space

For the ALOHA dataset, both `observation.state` and `action` are 14-dimensional vectors. The dimensions map to:

```
Dim 0:  left_waist           Dim 7:  right_waist
Dim 1:  left_shoulder        Dim 8:  right_shoulder
Dim 2:  left_elbow           Dim 9:  right_elbow
Dim 3:  left_forearm_roll    Dim 10: right_forearm_roll
Dim 4:  left_wrist_angle     Dim 11: right_wrist_angle
Dim 5:  left_wrist_rotate    Dim 12: right_wrist_rotate
Dim 6:  left_gripper         Dim 13: right_gripper
```

Joint values are in radians (roughly [-1.5, 1.5]). Gripper values are non-negative: 0 = closed, ~1.0 = open.

**Position control:** The action IS a target joint position, and the observation IS the current joint position. The robot's low-level controller handles moving from current to target. You can tell because state and action ranges are nearly identical, with action slightly leading state.

## How Video Decoding Works

When you access a frame's image (`dataset[i]["observation.images.top"]`), LeRobot doesn't load the entire video into memory. It:

1. Determines which episode and frame index you need
2. Opens the corresponding MP4 file
3. Uses torchcodec to seek to the exact frame
4. Decodes just that frame
5. Returns it as a float32 tensor normalised to [0, 1]

This lazy decoding is critical for large datasets — you never hold all video frames in memory.

## delta_timestamps: Getting Multiple Frames

A unique feature of LeRobotDataset is `delta_timestamps`. Instead of getting one frame at a time, you can configure the dataset to return multiple frames relative to the indexed frame:

```python
delta_timestamps = {
    "observation.image": [-1.0, -0.5, 0.0],  # current frame + 2 history frames
}
```

This returns frames from 1 second ago, 0.5 seconds ago, and now. Policies that need temporal context (seeing how the scene has been changing) use this to get history without manually managing frame buffers.

## Episode Boundaries

Episodes are contiguous blocks of frames. The `episode_index` column in the Parquet data tells you which frames belong to which episode. The `next.done` field is True only at the last frame of each episode.

In our ALOHA dataset, all 50 episodes have exactly 400 frames (fixed-length scripted demos). Real teleop data will have variable-length episodes — some demos take longer than others, and failed demos might be cut short.

## Loading a Dataset: What Happens Under the Hood

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
```

This single line:
1. Checks the local cache (`~/.cache/huggingface/lerobot/`)
2. If not cached, downloads from HuggingFace Hub — Parquet files and MP4 videos
3. Loads the Parquet data into a HuggingFace `Dataset` object (stored as `dataset.hf_dataset`)
4. Sets up video decoding pipelines for each camera feature
5. Exposes a PyTorch Dataset interface (supports `dataset[i]` and `len(dataset)`)

## Key Attributes of LeRobotDataset

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset.repo_id` | str | Hub repository ID |
| `dataset.meta` | LeRobotDatasetMetadata | Metadata object |
| `dataset.features` | dict | Feature definitions with shapes and dtypes |
| `dataset.fps` | int | Recording frame rate |
| `dataset.num_episodes` | int | Total number of episodes |
| `dataset.num_frames` | int | Total number of frames |
| `dataset.hf_dataset` | Dataset | Underlying HuggingFace Dataset |

Note: there is no `dataset.info` attribute (a common mistake). Use `dataset.meta` for metadata and `dataset.features` for feature definitions.

## The Inspection Script: What It Does

Our `src/inspect_dataset.py` profiles any LeRobotDataset:

1. **Metadata inspection** — reads structural info from `dataset.meta` and `dataset.features`
2. **Sample inspection** — loads individual frames, prints every field's shape, dtype, and value range
3. **Episode structure** — computes episode boundaries from `hf_dataset["episode_index"]`
4. **State/action range analysis** — samples across the dataset to compute per-joint statistics
5. **Frame extraction** — decodes sample video frames and saves image grids

The script is designed to be reusable on any LeRobotDataset — change `--repo-id` and it works.

## Common Gotchas

**Cache issues:** If you modify a dataset on the Hub, your local cache might not update. Delete `~/.cache/huggingface/lerobot/{repo_id}/` to force a fresh download.

**Video codec compatibility:** LeRobot requires ffmpeg for video decoding. If you get `torchcodec` errors, install ffmpeg 7.x.

**API changes across versions:** The LeRobot API has changed significantly between versions. Always inspect the actual object attributes rather than relying on documentation. Our inspection script ran into this — `dataset.info` didn't exist, we had to use `dataset.meta`.

**Parquet vs in-memory:** The Parquet files on disk are the source of truth. The `hf_dataset` is an in-memory representation. If something looks wrong, check the Parquet files directly with `pyarrow`.

## Summary

| Concept | Key Point |
|---------|-----------|
| Storage format | Parquet (numerical) + MP4 (video) |
| Frame access | `dataset[i]` returns a dict with all features |
| Video decoding | Lazy, one frame at a time via torchcodec |
| Action space | 14-dim: 7 joints per arm (6 joints + gripper) |
| Control type | Position control (action = target joint position) |
| Episode boundaries | Tracked via `episode_index` column |
| Metadata | `dataset.meta`, `dataset.features`, `dataset.fps` |

## What's Next

[Chapter 2: Data Analysis for Robot Learning →](02_data_analysis.md)