# Chapter 6: Diffusion Policy Deep-Dive

## The Problem

ACT handles multimodality through its CVAE — sampling different z values produces different action strategies. But the CVAE is a relatively weak form of multimodal modeling. The latent space is low-dimensional and the decoder is deterministic given z. For tasks with complex, high-dimensional multimodality (many valid trajectories that differ in subtle ways), ACT's CVAE may not capture the full distribution.

Diffusion Policy takes a fundamentally different approach borrowed from image generation (DALL-E, Stable Diffusion): model the action distribution by learning to iteratively denoise random noise into valid actions.

## How Diffusion Works: The Core Idea

Imagine you have a perfect demonstration action chunk — a [100, 14] tensor of 100 timesteps of 14 joint positions. Now imagine gradually adding Gaussian noise to it, step by step, until it's pure static. This is the **forward process** — turning signal into noise.

**Training** teaches a neural network to reverse this process: given a noisy action chunk at noise level t, predict what the slightly less noisy version looks like. This is called **denoising**.

**Inference** starts with pure random noise and iteratively denoises it into a valid action chunk. Each denoising step makes the actions slightly more coherent, slightly more like a real demonstration. After 100 denoising steps, what started as random numbers has become a smooth, physically plausible action sequence.

## The Forward Process (Adding Noise)

Given a clean action chunk $a_0$ from the dataset, the forward process produces increasingly noisy versions:

$$a_t = \sqrt{\bar{\alpha}_t} \cdot a_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

where $\epsilon$ is random Gaussian noise and $\bar{\alpha}_t$ is a noise schedule that controls how much noise is added at step $t$. At $t=0$, $\bar{\alpha}_0 \approx 1$ (almost clean). At $t=T$ (e.g., $T=100$), $\bar{\alpha}_T \approx 0$ (almost pure noise).

This is just a weighted blend between the original signal and noise, with the weights shifting from "mostly signal" to "mostly noise" as t increases.

## The Reverse Process (Removing Noise)

A neural network (the denoiser) learns to predict the noise $\epsilon$ that was added:

$$\hat{\epsilon} = f_\theta(a_t, t, \text{observation})$$

Given the noisy action $a_t$, the current noise level $t$, and the observation (image + joint state), the network predicts what noise was added. Subtracting this predicted noise gives a cleaner version of the action.

At inference, start with pure noise $a_T$ and iterate:
1. Predict noise: $\hat{\epsilon} = f_\theta(a_T, T, \text{obs})$
2. Remove predicted noise to get $a_{T-1}$
3. Predict noise: $\hat{\epsilon} = f_\theta(a_{T-1}, T-1, \text{obs})$
4. Remove predicted noise to get $a_{T-2}$
5. ... repeat 100 times ...
6. Final result: $a_0$ — a clean action chunk

## The Architecture: Conditional U-Net

The denoiser network is a **U-Net** — an architecture originally designed for image segmentation, adapted here for 1D action sequences.

The observation (image + state) is processed similarly to ACT: ResNet18 extracts visual features, the joint state is projected, and these are combined into a conditioning vector. This conditioning is injected into the U-Net at multiple scales via FiLM (Feature-wise Linear Modulation) — the observation modulates how the denoising operates.

The U-Net processes the noisy action chunk through:
1. **Downsampling blocks** — compress the temporal dimension, extracting coarse patterns
2. **Bottleneck** — lowest resolution, captures global structure
3. **Upsampling blocks** — expand back to original resolution, with skip connections from the downsampling path

The noise level $t$ is encoded as a sinusoidal embedding (similar to positional encoding in Transformers) and injected at every block.

## Why Diffusion Handles Multimodality

The key insight: the denoising process doesn't predict a single action — it starts from random noise and can converge to any of the valid modes in the training distribution. Different random noise samples lead to different denoising trajectories, which end at different modes.

Contrast with a regression model (predicting the mean): if there are two valid grasping strategies, regression averages them into an invalid strategy. Diffusion starts from random noise and collapses onto one strategy or the other, never the average.

This is the same mechanism that lets image diffusion models generate diverse images from the same text prompt — different noise seeds produce different valid images.

## Training Objective

The training loss is simple: mean squared error between the predicted noise and the actual noise:

$$L = ||\epsilon - f_\theta(a_t, t, \text{obs})||^2$$

For each training step:
1. Sample a clean action chunk from the dataset
2. Sample a random noise level $t$
3. Add noise to get $a_t$
4. Predict the noise: $\hat{\epsilon} = f_\theta(a_t, t, \text{obs})$
5. Loss = MSE between $\hat{\epsilon}$ and $\epsilon$

## Why Diffusion Needs More Training

ACT makes one forward pass and directly predicts actions. The mapping from observation to action is relatively direct — the Transformer learns it quickly.

Diffusion must learn to denoise at all 100 noise levels. At high noise (t near T), the network sees almost pure static and must predict the gross structure. At low noise (t near 0), the network sees almost clean actions and must predict fine-grained corrections. These are fundamentally different tasks that the same network must handle, parameterised by t.

This is why our Diffusion Policy at 35k steps (10% success) dramatically underperformed ACT at 40k steps (58% success). The denoising process hadn't converged — the network could partially denoise but not accurately enough to produce physically valid action sequences. Reference Diffusion Policy models train for 200k+ steps.

## Inference Speed: The Tradeoff

ACT: 1 forward pass → action chunk. Fast.

Diffusion: 100 forward passes (one per denoising step) → action chunk. Slow.

Our evaluation showed ACT at ~10 seconds per episode vs Diffusion at ~170 seconds. This 17x difference has real implications for deployment:

- At 50Hz control rate, ACT can predict action chunks much faster than needed — plenty of headroom
- Diffusion Policy at 100 denoising steps would struggle to maintain 50Hz in real-time
- Solutions: use fewer denoising steps (e.g., 10 instead of 100) with DDIM sampling, or predict less frequently with longer action chunks

## Our Results

| Run | Steps | Success Rate |
|-----|-------|-------------|
| Kaggle (bs16, 35k) | 35,000 | 10% |
| Reference (A100, 200k) | 200,000 | ~80% |

At 35k steps, the model was severely undertrained. The 10% success rate likely came from episodes where the initial conditions happened to be easy and the partially-trained denoising process got lucky.

## When to Use Diffusion Policy

**Use Diffusion when:**
- The task has genuinely multimodal solutions (multiple valid strategies)
- ACT's CVAE isn't capturing the diversity (policy always does the same thing)
- You can afford the training time (100k+ steps) and inference latency
- You need the highest possible task completion for complex manipulation

**Use ACT when:**
- You need fast iteration (converges in 20-40k steps)
- Inference speed matters (real-time control)
- The task has a relatively clear optimal strategy
- You're prototyping and need quick results

## Summary

| Concept | Key Point |
|---------|-----------|
| Forward process | Gradually add noise to clean actions |
| Reverse process | Iteratively denoise random noise into actions |
| U-Net denoiser | Conditioned on observation + noise level |
| Multimodality | Different noise seeds → different valid actions |
| Training objective | Predict the noise that was added (MSE loss) |
| Convergence | Needs 100k-200k+ steps (much more than ACT) |
| Inference speed | 100 forward passes per prediction (17x slower than ACT) |

## What's Next

[Chapter 7: Policy Comparison →](07_policy_comparison.md)# Chapter 6: Diffusion Policy Deep-Dive

## The Problem

ACT handles multimodality through its CVAE — sampling different z values produces different action strategies. But the CVAE is a relatively weak form of multimodal modeling. The latent space is low-dimensional and the decoder is deterministic given z. For tasks with complex, high-dimensional multimodality (many valid trajectories that differ in subtle ways), ACT's CVAE may not capture the full distribution.

Diffusion Policy takes a fundamentally different approach borrowed from image generation (DALL-E, Stable Diffusion): model the action distribution by learning to iteratively denoise random noise into valid actions.

## How Diffusion Works: The Core Idea

Imagine you have a perfect demonstration action chunk — a [100, 14] tensor of 100 timesteps of 14 joint positions. Now imagine gradually adding Gaussian noise to it, step by step, until it's pure static. This is the **forward process** — turning signal into noise.

**Training** teaches a neural network to reverse this process: given a noisy action chunk at noise level t, predict what the slightly less noisy version looks like. This is called **denoising**.

**Inference** starts with pure random noise and iteratively denoises it into a valid action chunk. Each denoising step makes the actions slightly more coherent, slightly more like a real demonstration. After 100 denoising steps, what started as random numbers has become a smooth, physically plausible action sequence.

## The Forward Process (Adding Noise)

Given a clean action chunk $a_0$ from the dataset, the forward process produces increasingly noisy versions:

$$a_t = \sqrt{\bar{\alpha}_t} \cdot a_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

where $\epsilon$ is random Gaussian noise and $\bar{\alpha}_t$ is a noise schedule that controls how much noise is added at step $t$. At $t=0$, $\bar{\alpha}_0 \approx 1$ (almost clean). At $t=T$ (e.g., $T=100$), $\bar{\alpha}_T \approx 0$ (almost pure noise).

This is just a weighted blend between the original signal and noise, with the weights shifting from "mostly signal" to "mostly noise" as t increases.

## The Reverse Process (Removing Noise)

A neural network (the denoiser) learns to predict the noise $\epsilon$ that was added:

$$\hat{\epsilon} = f_\theta(a_t, t, \text{observation})$$

Given the noisy action $a_t$, the current noise level $t$, and the observation (image + joint state), the network predicts what noise was added. Subtracting this predicted noise gives a cleaner version of the action.

At inference, start with pure noise $a_T$ and iterate:
1. Predict noise: $\hat{\epsilon} = f_\theta(a_T, T, \text{obs})$
2. Remove predicted noise to get $a_{T-1}$
3. Predict noise: $\hat{\epsilon} = f_\theta(a_{T-1}, T-1, \text{obs})$
4. Remove predicted noise to get $a_{T-2}$
5. ... repeat 100 times ...
6. Final result: $a_0$ — a clean action chunk

## The Architecture: Conditional U-Net

The denoiser network is a **U-Net** — an architecture originally designed for image segmentation, adapted here for 1D action sequences.

The observation (image + state) is processed similarly to ACT: ResNet18 extracts visual features, the joint state is projected, and these are combined into a conditioning vector. This conditioning is injected into the U-Net at multiple scales via FiLM (Feature-wise Linear Modulation) — the observation modulates how the denoising operates.

The U-Net processes the noisy action chunk through:
1. **Downsampling blocks** — compress the temporal dimension, extracting coarse patterns
2. **Bottleneck** — lowest resolution, captures global structure
3. **Upsampling blocks** — expand back to original resolution, with skip connections from the downsampling path

The noise level $t$ is encoded as a sinusoidal embedding (similar to positional encoding in Transformers) and injected at every block.

## Why Diffusion Handles Multimodality

The key insight: the denoising process doesn't predict a single action — it starts from random noise and can converge to any of the valid modes in the training distribution. Different random noise samples lead to different denoising trajectories, which end at different modes.

Contrast with a regression model (predicting the mean): if there are two valid grasping strategies, regression averages them into an invalid strategy. Diffusion starts from random noise and collapses onto one strategy or the other, never the average.

This is the same mechanism that lets image diffusion models generate diverse images from the same text prompt — different noise seeds produce different valid images.

## Training Objective

The training loss is simple: mean squared error between the predicted noise and the actual noise:

$$L = ||\epsilon - f_\theta(a_t, t, \text{obs})||^2$$

For each training step:
1. Sample a clean action chunk from the dataset
2. Sample a random noise level $t$
3. Add noise to get $a_t$
4. Predict the noise: $\hat{\epsilon} = f_\theta(a_t, t, \text{obs})$
5. Loss = MSE between $\hat{\epsilon}$ and $\epsilon$

## Why Diffusion Needs More Training

ACT makes one forward pass and directly predicts actions. The mapping from observation to action is relatively direct — the Transformer learns it quickly.

Diffusion must learn to denoise at all 100 noise levels. At high noise (t near T), the network sees almost pure static and must predict the gross structure. At low noise (t near 0), the network sees almost clean actions and must predict fine-grained corrections. These are fundamentally different tasks that the same network must handle, parameterised by t.

This is why our Diffusion Policy at 35k steps (10% success) dramatically underperformed ACT at 40k steps (58% success). The denoising process hadn't converged — the network could partially denoise but not accurately enough to produce physically valid action sequences. Reference Diffusion Policy models train for 200k+ steps.

## Inference Speed: The Tradeoff

ACT: 1 forward pass → action chunk. Fast.

Diffusion: 100 forward passes (one per denoising step) → action chunk. Slow.

Our evaluation showed ACT at ~10 seconds per episode vs Diffusion at ~170 seconds. This 17x difference has real implications for deployment:

- At 50Hz control rate, ACT can predict action chunks much faster than needed — plenty of headroom
- Diffusion Policy at 100 denoising steps would struggle to maintain 50Hz in real-time
- Solutions: use fewer denoising steps (e.g., 10 instead of 100) with DDIM sampling, or predict less frequently with longer action chunks

## Our Results

| Run | Steps | Success Rate |
|-----|-------|-------------|
| Kaggle (bs16, 35k) | 35,000 | 10% |
| Reference (A100, 200k) | 200,000 | ~80% |

At 35k steps, the model was severely undertrained. The 10% success rate likely came from episodes where the initial conditions happened to be easy and the partially-trained denoising process got lucky.

## When to Use Diffusion Policy

**Use Diffusion when:**
- The task has genuinely multimodal solutions (multiple valid strategies)
- ACT's CVAE isn't capturing the diversity (policy always does the same thing)
- You can afford the training time (100k+ steps) and inference latency
- You need the highest possible task completion for complex manipulation

**Use ACT when:**
- You need fast iteration (converges in 20-40k steps)
- Inference speed matters (real-time control)
- The task has a relatively clear optimal strategy
- You're prototyping and need quick results

## Summary

| Concept | Key Point |
|---------|-----------|
| Forward process | Gradually add noise to clean actions |
| Reverse process | Iteratively denoise random noise into actions |
| U-Net denoiser | Conditioned on observation + noise level |
| Multimodality | Different noise seeds → different valid actions |
| Training objective | Predict the noise that was added (MSE loss) |
| Convergence | Needs 100k-200k+ steps (much more than ACT) |
| Inference speed | 100 forward passes per prediction (17x slower than ACT) |

## What's Next

[Chapter 7: Policy Comparison →](07_policy_comparison.md)