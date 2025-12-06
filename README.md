# Custom GRPO - Group Relative Policy Optimization

This project implements a learning framework for GRPO (Group Relative Policy Optimization) on the GSM8K dataset. The core GRPO algorithm implementation is left as a placeholder for you to learn and implement.

## Project Structure

```
.
├── main.py              # Main training script
├── grpo_trainer.py      # GRPO trainer with placeholder implementation
├── config.py            # Configuration dataclass
├── config/              # Hydra configuration directory
│   ├── config.yaml      # Main config file
│   ├── model/           # Model configurations
│   ├── training/        # Training configurations
│   ├── grpo/            # GRPO-specific configurations
│   └── wandb/           # WandB configurations
└── README.md           # This file
```

## Overview

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm for fine-tuning language models. It optimizes policies by comparing responses within groups and using relative preferences, which can be more stable and effective than absolute reward-based methods.

## Key Components

### 1. Data Loading (`main.py`)
- Loads GSM8K dataset from HuggingFace
- Formats prompts and answers for training
- Creates train/test splits

### 2. Model Setup (`main.py`)
- Uses `AutoModelForCausalLM` from transformers
- Supports any causal language model (GPT-2, GPT-Neo, LLaMA, etc.)
- Handles tokenization and batching

### 3. GRPO Trainer (`grpo_trainer.py`)
The `GRPOTrainer` class contains placeholder methods for you to implement:

- **`train_step()`**: Main training step - implement the GRPO algorithm here
- **`compute_grpo_loss()`**: Compute the GRPO loss with group relative comparisons
- **`compute_rewards()`**: Compute rewards for generated responses
- **`generate_responses()`**: Generate multiple responses per prompt
- **`form_groups()`**: Form groups for relative comparisons

### 4. Evaluation (`main.py`)
- Evaluates model accuracy on GSM8K test set
- Extracts numerical answers and compares with ground truth
- Tracks rewards using WandB

## Configuration with Hydra

This project uses [Hydra](https://hydra.cc/) for configuration management. Configurations are organized in the `config/` directory.

### Basic Usage

```bash
# Use default configuration
python main.py

# Override specific parameters from command line
python main.py training.batch_size=8 training.learning_rate=2e-5

# Use a different training config
python main.py training=large_batch

# Enable WandB logging
python main.py wandb=enabled wandb.project=my-project

# Combine multiple overrides
python main.py training=large_batch wandb=enabled wandb.run_name=experiment-1
```

### Configuration Structure

The configuration is organized into groups:

- **`model`**: Model settings (model name, max length)
  - Available: `gpt2` (default)
  
- **`training`**: Training hyperparameters
  - Available: `default`, `large_batch`
  
- **`grpo`**: GRPO-specific settings
  - Available: `default`
  
- **`wandb`**: WandB logging settings
  - Available: `default`, `enabled`

### Example: Creating Custom Configs

You can create custom configurations by adding new YAML files:

**`config/training/custom.yaml`**:
```yaml
# @package training

batch_size: 32
learning_rate: 5e-6
num_epochs: 10
max_grad_norm: 0.5
weight_decay: 0.01
```

Then use it with:
```bash
python main.py training=custom
```

### Command Line Overrides

You can override any config value from the command line:

```bash
# Override nested values
python main.py model.model_name=gpt2-medium training.batch_size=16

# Override top-level values
python main.py output_dir=./my_outputs

# Override dataset settings
python main.py dataset.max_eval_samples=200
```

### Hydra Outputs

Hydra automatically creates output directories for each run:
- `outputs/YYYY-MM-DD/HH-MM-SS/` - Contains logs and configs
- Your specified `output_dir` - Contains model checkpoints

## Implementation Guide

### Step 1: Understand GRPO
GRPO typically involves:
1. Generating multiple responses for each prompt
2. Computing rewards for each response
3. Grouping responses and computing relative preferences
4. Computing policy loss using relative comparisons
5. Adding KL penalty to prevent policy deviation (optional)

### Step 2: Implement Reward Computation
In `compute_rewards()`, implement reward calculation based on:
- Exact match with ground truth
- Numerical answer correctness
- Step-by-step reasoning quality (optional)

### Step 3: Implement Group Formation
In `form_groups()`, decide how to group responses:
- Random assignment
- Similarity-based grouping
- Fixed batch structure
- Dynamic grouping based on rewards

### Step 4: Implement GRPO Loss
In `compute_grpo_loss()`, implement:
- Relative preference computation within groups
- Policy gradient with relative advantages
- KL divergence penalty (if using reference model)

### Step 5: Complete Training Step
In `train_step()`, wire everything together:
- Generate responses
- Compute rewards
- Form groups
- Compute GRPO loss
- Backpropagate and update

## Usage Examples

### Basic Training
```bash
python main.py
```

### Training with WandB
```bash
python main.py wandb=enabled wandb.project=grpo-experiments
```

### Large Batch Training
```bash
python main.py training=large_batch
```

### Custom Configuration
```bash
python main.py \
    training.batch_size=16 \
    training.learning_rate=2e-5 \
    grpo.group_size=8 \
    wandb=enabled \
    wandb.run_name=my-experiment
```

## Dependencies

Assumes the following packages are installed:
- `torch`
- `transformers`
- `datasets`
- `tqdm`
- `wandb` (optional, for logging)
- `hydra-core` (for configuration management)
- `omegaconf` (dependency of Hydra)

## WandB Integration

The project includes WandB integration for tracking:
- Training loss and GRPO loss
- Evaluation accuracy
- **Reward statistics** (mean, std, min, max) on the dataset
- Best model accuracy

To enable WandB:
```bash
python main.py wandb=enabled
```

Or override specific WandB settings:
```bash
python main.py wandb=enabled wandb.project=my-project wandb.run_name=exp-1
```

## Learning Resources

To understand GRPO better, consider:
1. Reading about policy gradient methods (REINFORCE, PPO)
2. Understanding relative preference learning
3. Studying group-based optimization strategies
4. Reviewing RLHF (Reinforcement Learning from Human Feedback) literature

## Notes

- The current implementation uses a simple supervised learning loss as a placeholder
- Replace the placeholder code in `grpo_trainer.py` with your GRPO implementation
- The evaluation metric is simple accuracy - you may want to add more sophisticated metrics
- Consider implementing a reference model for KL penalty if needed
- Hydra automatically saves the config used for each run in the output directory

## Next Steps

1. Implement `compute_rewards()` with your reward function
2. Implement `form_groups()` with your grouping strategy
3. Implement `compute_grpo_loss()` with the GRPO algorithm
4. Complete `train_step()` to wire everything together
5. Experiment with hyperparameters and group formation strategies
6. Use Hydra's multirun feature for hyperparameter sweeps: `python main.py -m training.batch_size=4,8,16`

Good luck learning GRPO! 🚀
