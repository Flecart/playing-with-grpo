"""
GRPO (Group Relative Policy Optimization) Trainer.

This module contains the GRPO trainer class with placeholder methods
for the core GRPO algorithm implementation.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import TrainingConfig


class GRPOTrainer:
    """
    Trainer for Group Relative Policy Optimization (GRPO).
    
    GRPO is a reinforcement learning algorithm that optimizes language models
    by comparing responses in groups and using relative preferences.
    
    The core GRPO algorithm implementation is left as a placeholder for learning.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler (will be set up in training loop if needed)
        self.scheduler = None
        
        # Reference model for KL divergence (frozen copy of initial model)
        # In GRPO, you might want to keep a reference model
        self.reference_model = None
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step using GRPO.
        
        Args:
            batch: Dictionary containing:
                - input_ids: Tokenized prompts [batch_size, seq_len]
                - attention_mask: Attention masks [batch_size, seq_len]
                - full_input_ids: Tokenized prompts + responses [batch_size, seq_len]
                - full_attention_mask: Attention masks for full sequences
                - question: Original questions (list of strings)
                - answer: Ground truth answers (list of strings)
        
        Returns:
            Dictionary with loss values and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Extract batch components
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        full_input_ids = batch["full_input_ids"].to(self.device)
        full_attention_mask = batch["full_attention_mask"].to(self.device)
        questions = batch["question"]
        answers = batch["answer"]
        
        batch_size = input_ids.shape[0]
        
        # ============================================================
        # TODO: Implement GRPO algorithm here
        # ============================================================
        # 
        # GRPO typically involves:
        # 1. Generate multiple responses for each prompt (or use provided responses)
        # 2. Compute rewards for each response (using a reward model or ground truth)
        # 3. Group responses and compute relative preferences
        # 4. Compute policy loss using relative comparisons
        # 5. Add KL penalty to prevent policy from deviating too much
        # 6. Backpropagate and update
        #
        # Key components you'll need to implement:
        # - Response generation (sampling from policy)
        # - Reward computation (could use ground truth for GSM8K)
        # - Group formation (how to group responses for comparison)
        # - Relative preference computation
        # - Policy gradient loss with relative comparisons
        # - KL divergence penalty (optional, depends on GRPO variant)
        #
        # Example structure:
        # 1. Get logits from model for full sequences
        # 2. Compute log probabilities
        # 3. Generate or use responses
        # 4. Compute rewards (e.g., based on answer correctness)
        # 5. Form groups and compute relative preferences
        # 6. Compute GRPO loss
        # 7. Backward pass
        #
        # ============================================================
        
        # PLACEHOLDER: Basic supervised learning loss (replace with GRPO)
        # This is just to make the code run - you should replace this!
        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
        )
        
        # Shift labels for next token prediction
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = full_input_ids[:, 1:].contiguous()
        shift_mask = full_attention_mask[:, 1:].contiguous()
        
        # Compute cross-entropy loss only on response tokens (not prompt tokens)
        prompt_length = input_ids.shape[1]
        response_logits = shift_logits[:, prompt_length - 1:, :]
        response_labels = shift_labels[:, prompt_length - 1:]
        response_mask = shift_mask[:, prompt_length - 1:]
        
        loss = F.cross_entropy(
            response_logits.reshape(-1, response_logits.size(-1)),
            response_labels.reshape(-1),
            reduction="none",
        )
        loss = (loss * response_mask.reshape(-1)).sum() / response_mask.sum()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute rewards for logging (optional, can be expensive)
        # You might want to compute this less frequently
        metrics = {
            "loss": loss.item(),
            "grpo_loss": loss.item(),  # Replace with actual GRPO loss
        }
        
        # Optionally compute and include reward metrics
        # This can be expensive, so you might want to do it less frequently
        # Uncomment if you want to track rewards during training:
        # try:
        #     prompts = [f"Question: {q}\nAnswer:" for q in questions]
        #     generated_responses = self.generate_responses(
        #         prompts,
        #         num_samples=1,
        #         max_new_tokens=256,
        #     )
        #     flat_responses = [resp[0] for resp in generated_responses]
        #     rewards = self.compute_rewards(flat_responses, answers)
        #     metrics["mean_reward"] = rewards.mean().item()
        #     metrics["std_reward"] = rewards.std().item()
        # except Exception as e:
        #     # Skip reward computation if it fails
        #     pass
        
        return metrics
    
    def compute_rewards(
        self,
        responses: List[str],
        ground_truth_answers: List[str],
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        
        For GSM8K, rewards can be based on:
        - Exact match with ground truth
        - Numerical answer correctness
        - Step-by-step reasoning quality
        
        Args:
            responses: List of generated response strings
            ground_truth_answers: List of ground truth answer strings
        
        Returns:
            Tensor of rewards [batch_size]
        """
        # ============================================================
        # TODO: Implement reward computation
        # ============================================================
        # 
        # For GSM8K, you might want to:
        # 1. Extract numerical answers from responses
        # 2. Compare with ground truth answers
        # 3. Assign rewards (e.g., 1.0 for correct, 0.0 for incorrect)
        # 4. Optionally: reward partial credit for correct reasoning steps
        #
        # ============================================================
        
        rewards = []
        for response, gt_answer in zip(responses, ground_truth_answers):
            # Simple placeholder: extract numbers and compare
            pred_num = self._extract_answer_number(response)
            gt_num = self._extract_answer_number(gt_answer)
            
            if pred_num is not None and gt_num is not None:
                if abs(pred_num - gt_num) < 1e-6:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def _extract_answer_number(self, text: str) -> Optional[float]:
        """Extract the final numerical answer from text."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                return None
        return None
    
    def generate_responses(
        self,
        prompts: List[str],
        num_samples: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> List[List[str]]:
        """
        Generate responses for given prompts.
        
        Args:
            prompts: List of prompt strings
            num_samples: Number of samples to generate per prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
        
        Returns:
            List of lists of generated responses [batch_size, num_samples]
        """
        self.model.eval()
        all_responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                prompt_responses = []
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                for _ in range(num_samples):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    generated = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    prompt_responses.append(generated)
                
                all_responses.append(prompt_responses)
        
        self.model.train()
        return all_responses
    
    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        groups: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss using group relative comparisons.
        
        Args:
            log_probs: Log probabilities of responses [batch_size] or [batch_size, num_samples]
            rewards: Rewards for responses [batch_size] or [batch_size, num_samples]
            groups: Optional list of groups. Each group is a list of indices.
                   If None, will form groups automatically.
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # ============================================================
        # TODO: Implement GRPO loss computation
        # ============================================================
        # 
        # GRPO loss typically involves:
        # 1. Forming groups of responses (if not provided)
        # 2. Computing relative preferences within groups
        # 3. Computing policy gradient with relative comparisons
        # 4. Adding KL penalty (optional)
        #
        # The key idea is to compare responses within groups rather than
        # using absolute rewards, which can be more stable and effective.
        #
        # Example structure:
        # - Normalize rewards within groups
        # - Compute relative advantages
        # - Weight log probabilities by relative advantages
        # - Add KL divergence penalty if using reference model
        #
        # ============================================================
        
        # PLACEHOLDER: Simple policy gradient (replace with GRPO!)
        # This is just a placeholder - implement the actual GRPO algorithm
        advantages = rewards - rewards.mean()
        loss = -(log_probs * advantages).mean()
        
        metrics = {
            "grpo_loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_log_prob": log_probs.mean().item(),
        }
        
        return loss, metrics
    
    def form_groups(
        self,
        batch_size: int,
        group_size: int = 4,
    ) -> List[List[int]]:
        """
        Form groups of indices for group-based comparisons.
        
        Args:
            batch_size: Total number of samples
            group_size: Desired size of each group
        
        Returns:
            List of groups, where each group is a list of indices
        """
        # ============================================================
        # TODO: Implement group formation strategy
        # ============================================================
        # 
        # You might want to:
        # - Randomly assign samples to groups
        # - Group by similarity (e.g., similar prompts)
        # - Use fixed groups based on batch structure
        # - Dynamic grouping based on rewards
        #
        # ============================================================
        
        # Simple placeholder: sequential grouping
        groups = []
        for i in range(0, batch_size, group_size):
            group = list(range(i, min(i + group_size, batch_size)))
            if len(group) > 1:  # Only form groups with at least 2 members
                groups.append(group)
        
        return groups
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        step: int,
    ):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "step": step,
            },
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )
        
        print(f"Saved checkpoint to {checkpoint_dir}")

