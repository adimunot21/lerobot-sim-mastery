# Chapter 3: ACT Policy Deep-Dive

## The Problem

You have 50 demonstrations of a robot transferring a cube. Each demonstration is a sequence of (observation, action) pairs. You need a neural network that takes the current observation and outputs the correct action. This is **imitation learning** — learning to imitate the demonstrator's behaviour.

The simplest approach is behaviour cloning: train a network to predict `action` given `observation` using supervised learning. But naive behaviour cloning has a fundamental problem: **compounding errors**. A small mistake at step t shifts the robot to a state slightly different from anything in the training data. At step t+1, the prediction is slightly worse because the input is unfamiliar. By step t+50, the robot is completely off-track.

ACT (Action Chunking Transformer) solves this with two key ideas: action chunking and a Transformer architecture with a CVAE.

## Idea 1: Action Chunking

Instead of predicting one action at a time, ACT predicts a **chunk** of future actions — typically 100 timesteps into the future. The model outputs a sequence: "here's what the robot should do for the next 100 frames."

Why this helps: by committing to a long-horizon plan, the model avoids the compounding error problem. Each prediction covers 100 steps, so errors only compound across chunk boundaries (every 100 steps) rather than every single step.

During execution, only the first `n_action_steps` actions from the chunk are actually sent to the robot (e.g., the first 100). Then a new chunk is predicted from the updated observation. In some implementations, overlapping chunks are blended using **temporal ensembling** — averaging the predictions from the current chunk and the previous chunk in the overlap region. This produces smoother transitions between chunks.

**Worked example:** At step 0, the model predicts actions for steps 0-99. The robot executes steps 0-99. At step 100, the model observes the new state and predicts actions for steps 100-199. If using temporal ensembling with overlap, at step 50 the model might predict a new chunk for steps 50-149, and steps 50-99 get the average of both predictions.

## Idea 2: The Transformer Architecture

ACT uses an encoder-decoder Transformer. Here's the data flow:

### Encoder (processes observations)

1. **Camera image** (480×640×3) → ResNet18 backbone → visual feature vector. ResNet18 is a convolutional neural network pre-trained on ImageNet that extracts spatial features. The raw pixels become a compact representation capturing "what's in the scene" — object positions, arm configuration, gripper state — without the model needing to reason about individual pixels.

2. **Joint state** (14-dim vector) → linear projection → state embedding. A simple matrix multiplication that maps the joint positions into the same dimensional space as the visual features.

3. Visual features + state embedding → **Transformer encoder** with self-attention. Self-attention lets different parts of the observation "talk to each other" — the model learns that seeing the cube at position X is relevant to what the arm at position Y should do.

### Decoder (predicts actions)

1. **Action queries** — one learnable vector per future timestep in the chunk (100 queries for a 100-step chunk). These start as random learned embeddings.

2. Each query **cross-attends** to the encoder output. Cross-attention means: "given what I know about the observation, what should the action be at my particular future timestep?" Each query specializes in a different point in the future.

3. Each query passes through a feedforward network → **14-dimensional action prediction**.

4. Output: a [100, 14] tensor — 100 timesteps, each with 14 joint target positions.

## Idea 3: The CVAE (Conditional Variational Autoencoder)

A plain Transformer decoder is deterministic — given the same observation, it always predicts the same action chunk. But manipulation tasks often have multiple valid solutions. You could approach the cube from the left or the right. You could grasp it with a pinch or a power grip.

The CVAE adds a latent variable **z** that captures the "style" of the action:

**During training:** The encoder also receives the ground-truth future actions (the full action chunk from the demonstration). It encodes them into a distribution over z. A sample from this distribution is fed to the decoder alongside the observation features. The model learns: "given this observation AND this style (z), produce these actions."

**During inference:** There are no ground-truth actions. Instead, z is sampled from the prior distribution (a standard Gaussian). Different z samples produce different valid action sequences. This is how ACT handles multimodality — it doesn't average between strategies, it commits to one based on the sampled z.

The training objective includes a **KL divergence** term that keeps the learned z distribution close to the Gaussian prior, ensuring that sampling from the prior at inference time produces reasonable z values.

## The Training Loop

Each training step:

1. **Sample a batch** of frames from the dataset (batch_size=32 on our A40 run). For each frame, load the camera image, joint state, and the next 100 ground-truth actions.

2. **Forward pass:** Encode the observation (ResNet18 + Transformer encoder). Encode the ground-truth actions into z (CVAE encoder). Decode the action chunk (Transformer decoder conditioned on z and observation).

3. **Compute loss:** L1 loss (mean absolute error) between predicted and ground-truth action chunks, plus KL divergence on z. L1 is preferred over L2 (mean squared error) because L2 over-penalises large errors, pushing the model to predict the mean of multimodal distributions rather than committing to one mode.

4. **Backpropagate:** Compute gradients of the loss with respect to every model parameter using PyTorch's autograd.

5. **Update weights:** The AdamW optimizer applies the gradients with weight decay regularization.

This repeats for 80k-100k steps. The model has ~50M parameters (mostly in ResNet18 and the Transformer layers), producing a ~207MB checkpoint file.

## Our Results

| Run | Steps | Batch Size | GPU | Success Rate |
|-----|-------|------------|-----|-------------|
| Local | 80k | 4 | GTX 1650 | 44% |
| Kaggle | 40k | 16 | T4 | 58% |
| A40 | 100k | 32 | A40 | 52% |
| Reference | 80k | 8 | A100 | ~90% |

ACT converges quickly — the loss drops sharply in the first 10-20k steps and then gradually flattens. The success rate plateaus around 52-58% in our runs, regardless of additional compute. The gap to the reference 90% is likely due to hyperparameter differences (learning rate schedule, chunk size configuration) rather than training budget.

## What the Parameters Mean

| Parameter | Our Value | What It Controls |
|-----------|-----------|-----------------|
| `batch_size` | 4-32 | Samples per gradient update. Larger = more stable gradients |
| `steps` | 80k-100k | Total optimization steps |
| `chunk_size` | 100 | How many future actions to predict |
| `n_action_steps` | 100 | How many actions to execute from each chunk |
| `n_obs_steps` | 1 | How many past observation frames to use |
| Learning rate | ~1e-5 | Step size for weight updates |

## Common Gotchas

**VRAM determines batch_size:** 4GB VRAM → batch_size=4. 16GB → batch_size=16. 48GB → batch_size=32. This is the most impactful constraint.

**Batch size matters more than steps:** Our Kaggle run (bs16, 40k steps) beat our local run (bs4, 80k steps). Stable gradients from larger batches are more valuable than extra training iterations.

**Loss doesn't tell the whole story:** Loss can still be decreasing while success rate has plateaued. Always evaluate checkpoints in simulation, don't rely solely on training loss.

## Summary

| Concept | Key Point |
|---------|-----------|
| Action chunking | Predict 100 future actions at once, reduces compounding errors |
| Transformer encoder | Processes image + state with self-attention |
| Transformer decoder | Cross-attends to observation, predicts action chunk |
| CVAE | Latent variable z captures action "style", handles multimodality |
| L1 loss | Preferred over L2 for multimodal action distributions |
| Fast convergence | Meaningful results within 20-40k steps |

## What's Next

[Chapter 4: Training Practicalities →](04_training_on_colab.md)