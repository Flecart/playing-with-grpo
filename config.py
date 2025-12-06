"""
Configuration classes for GRPO training.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    
    # Model settings
    model_name: str = "gpt2"
    max_length: int = 512
    
    # Training settings
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # GRPO-specific settings
    group_size: int = 4  # Size of groups for relative comparisons
    num_samples_per_prompt: int = 1  # Number of responses to generate per prompt
    temperature: float = 1.0  # Sampling temperature
    
    # KL penalty (if using reference model)
    kl_coeff: float = 0.1  # Coefficient for KL divergence penalty
    use_kl_penalty: bool = False  # Whether to use KL penalty
    
    # Evaluation and saving
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./outputs"
    
    # Generation settings
    max_new_tokens: int = 256
    do_sample: bool = True
    
    # Reward settings
    reward_type: str = "exact_match"  # "exact_match", "numerical", "step_by_step"
    
    # Logging
    logging_steps: int = 10
    log_dir: Optional[str] = None

