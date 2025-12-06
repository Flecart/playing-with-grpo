"""
Main training script for GRPO (Group Relative Policy Optimization) on GSM8K dataset.
"""

import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from grpo_trainer import GRPOTrainer
from config import TrainingConfig


class GSM8KDataset(Dataset):
    """Dataset wrapper for GSM8K math problems."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        split: str = "train",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]
        
        # Format the prompt
        prompt = f"Question: {question}\nAnswer:"
        full_text = f"{prompt} {answer}"
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        
        full_tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        
        return {
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "full_input_ids": full_tokens["input_ids"].squeeze(0),
            "full_attention_mask": full_tokens["attention_mask"].squeeze(0),
            "question": question,
            "answer": answer,
        }


def load_gsm8k_dataset(split: str = "train") -> List[Dict]:
    """Load GSM8K dataset from HuggingFace."""
    print(f"Loading GSM8K {split} dataset...")
    dataset = load_dataset("gsm8k", "main", split=split)
    data = []
    for item in dataset:
        data.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    print(f"Loaded {len(data)} examples from GSM8K {split}")
    return data


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict],
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate the model on GSM8K test set."""
    model.eval()
    correct = 0
    total = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for item in tqdm(eval_data[:100]):  # Evaluate on first 100 for speed
            question = item["question"]
            ground_truth = item["answer"]
            
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Simple accuracy check: extract number from answer
            try:
                # Extract final number from ground truth
                gt_number = extract_answer_number(ground_truth)
                pred_number = extract_answer_number(generated)
                
                if gt_number is not None and pred_number is not None:
                    if abs(gt_number - pred_number) < 1e-6:
                        correct += 1
                total += 1
            except Exception:
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def extract_answer_number(text: str) -> Optional[float]:
    """Extract the final numerical answer from text."""
    import re
    # Find all numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except (ValueError, IndexError):
            return None
    return None


def compute_dataset_rewards(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    trainer: GRPOTrainer,
    max_samples: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute average reward on a dataset subset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        dataset: List of dataset examples
        trainer: GRPOTrainer instance (for reward computation)
        max_samples: Maximum number of samples to evaluate
        device: Device to run on
    
    Returns:
        Dictionary with reward statistics
    """
    model.eval()
    all_rewards = []
    
    print(f"Computing rewards on {min(max_samples, len(dataset))} samples...")
    with torch.no_grad():
        for item in tqdm(dataset[:max_samples]):
            question = item["question"]
            ground_truth = item["answer"]
            
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            # Compute reward using trainer's reward function
            reward = trainer.compute_rewards(
                [generated],
                [ground_truth],
            )
            all_rewards.append(reward.item())
    
    model.train()
    
    rewards_tensor = torch.tensor(all_rewards)
    return {
        "mean_reward": rewards_tensor.mean().item(),
        "std_reward": rewards_tensor.std().item(),
        "min_reward": rewards_tensor.min().item(),
        "max_reward": rewards_tensor.max().item(),
        "reward_sum": rewards_tensor.sum().item(),
        "num_samples": len(all_rewards),
    }


def create_training_config_from_hydra(cfg: DictConfig) -> TrainingConfig:
    """Convert Hydra config to TrainingConfig dataclass."""
    return TrainingConfig(
        model_name=cfg.model.model_name,
        max_length=cfg.model.max_length,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        num_epochs=cfg.training.num_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        weight_decay=cfg.training.weight_decay,
        group_size=cfg.grpo.group_size,
        num_samples_per_prompt=cfg.grpo.num_samples_per_prompt,
        temperature=cfg.grpo.temperature,
        kl_coeff=cfg.grpo.kl_coeff,
        use_kl_penalty=cfg.grpo.use_kl_penalty,
        max_new_tokens=cfg.grpo.max_new_tokens,
        do_sample=cfg.grpo.do_sample,
        reward_type=cfg.grpo.reward_type,
        eval_steps=cfg.eval.eval_steps,
        save_steps=cfg.eval.save_steps,
        output_dir=cfg.output_dir,
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""
    
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Convert Hydra config to TrainingConfig
    config = create_training_config_from_hydra(cfg)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if cfg.wandb.use_wandb:
        wandb_config = {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "group_size": config.group_size,
            "num_samples_per_prompt": config.num_samples_per_prompt,
            "kl_coeff": config.kl_coeff,
        }
        # Add full config to wandb
        wandb_config.update(OmegaConf.to_container(cfg, resolve=True))
        
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name if cfg.wandb.run_name is not None else None,
            entity=cfg.wandb.entity if cfg.wandb.entity is not None else None,
            config=wandb_config,
        )
        print("Initialized wandb logging")
    
    # Set device
    if cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.to(device)
    
    # Load datasets
    train_data = load_gsm8k_dataset(split=cfg.dataset.split_train)
    eval_data = load_gsm8k_dataset(split=cfg.dataset.split_test)
    
    # Create datasets
    train_dataset = GSM8KDataset(
        train_data,
        tokenizer,
        max_length=config.max_length,
        split="train",
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_accuracy = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            # Train step
            loss_dict = trainer.train_step(batch)
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix(loss_dict)
            
            # Log to wandb
            if cfg.wandb.use_wandb:
                log_dict = {
                    "train/loss": loss_dict.get("loss", 0.0),
                    "train/grpo_loss": loss_dict.get("grpo_loss", 0.0),
                    "train/step": global_step,
                    "train/epoch": epoch + 1,
                }
                # Add any additional metrics from loss_dict
                for key, value in loss_dict.items():
                    if key not in ["loss", "grpo_loss"]:
                        log_dict[f"train/{key}"] = value
                wandb.log(log_dict)
            
            # Evaluation
            if global_step % config.eval_steps == 0:
                eval_results = evaluate_model(
                    model,
                    tokenizer,
                    eval_data,
                    max_new_tokens=cfg.eval.max_new_tokens,
                    device=str(device),
                )
                print(f"\nStep {global_step} - Eval Accuracy: {eval_results['accuracy']:.4f}")
                
                # Compute rewards on dataset
                reward_stats = compute_dataset_rewards(
                    model,
                    tokenizer,
                    train_data,
                    trainer,
                    max_samples=cfg.dataset.max_reward_eval_samples,
                    device=str(device),
                )
                print(f"Step {global_step} - Mean Reward: {reward_stats['mean_reward']:.4f}")
                
                # Log to wandb
                if cfg.wandb.use_wandb:
                    wandb.log({
                        "eval/accuracy": eval_results["accuracy"],
                        "eval/correct": eval_results["correct"],
                        "eval/total": eval_results["total"],
                        "reward/mean": reward_stats["mean_reward"],
                        "reward/std": reward_stats["std_reward"],
                        "reward/min": reward_stats["min_reward"],
                        "reward/max": reward_stats["max_reward"],
                        "reward/sum": reward_stats["reward_sum"],
                        "reward/num_samples": reward_stats["num_samples"],
                        "step": global_step,
                    })
                
                # Save best model
                if eval_results["accuracy"] > best_accuracy:
                    best_accuracy = eval_results["accuracy"]
                    trainer.save_checkpoint(
                        os.path.join(config.output_dir, "best_model"),
                        global_step,
                    )
                    print(f"Saved best model with accuracy: {best_accuracy:.4f}")
                    
                    if cfg.wandb.use_wandb:
                        wandb.log({"eval/best_accuracy": best_accuracy, "step": global_step})
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                trainer.save_checkpoint(
                    os.path.join(config.output_dir, f"checkpoint-{global_step}"),
                    global_step,
                )
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_results = evaluate_model(
        model,
        tokenizer,
        eval_data,
        max_new_tokens=cfg.eval.max_new_tokens,
        device=str(device),
    )
    print(f"Final Accuracy: {final_results['accuracy']:.4f}")
    
    # Final reward computation
    final_reward_stats = compute_dataset_rewards(
        model,
        tokenizer,
        train_data,
        trainer,
        max_samples=cfg.dataset.max_final_reward_samples,
        device=str(device),
    )
    print(f"Final Mean Reward: {final_reward_stats['mean_reward']:.4f}")
    
    # Log final metrics to wandb
    if cfg.wandb.use_wandb:
        wandb.log({
            "final/accuracy": final_results["accuracy"],
            "final/mean_reward": final_reward_stats["mean_reward"],
            "final/std_reward": final_reward_stats["std_reward"],
            "step": global_step,
        })
    
    # Save final model
    trainer.save_checkpoint(
        os.path.join(config.output_dir, "final_model"),
        global_step,
    )
    
    if cfg.wandb.use_wandb:
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
