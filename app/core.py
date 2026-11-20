"""Core implementation for Student Feedback Summarizer with LoRA from scratch."""
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Milestone 1: Dataset of synthetic student feedback and summary pairs
# ==============================================================================

SYNTHETIC_FEEDBACK_DATA: List[Dict[str, Any]] = [
    {
        "id": 1,
        "category": "Pacing & Clarity",
        "feedback": "The professor covered dynamic programming way too fast in week 4. However, the recorded videos and lecture slides with step-by-step traces were very helpful for revision.",
        "summary": "Slow down lecture pacing on dynamic programming; maintain high-quality slide traces and recordings.",
    },
    {
        "id": 2,
        "category": "Assignments & Workload",
        "feedback": "Homework 3 was disproportionately long compared to other weeks, and the setup instructions for Docker were ambiguous, causing students to lose days debugging environment errors.",
        "summary": "Streamline Homework 3 workload and clarify Docker environment setup instructions.",
    },
    {
        "id": 3,
        "category": "Interactive Engagement",
        "feedback": "The live coding demonstrations in class were fantastic and made complex pointer concepts click. More interactive classroom polls and coding exercises would be great.",
        "summary": "Expand live coding sessions and interactive polls to reinforce challenging concepts.",
    },
    {
        "id": 4,
        "category": "Grading & Feedback",
        "feedback": "We didn't receive feedback on the midterm project until week 11, long after the drop deadline. Furthermore, the grading rubric wasn't posted beforehand.",
        "summary": "Publish grading rubrics in advance and accelerate turnaround time for project feedback.",
    },
    {
        "id": 5,
        "category": "Course Materials",
        "feedback": "The textbook assigned for NLP algorithms feels outdated. It focuses heavily on rule-based parsers and barely mentions modern transformer architectures.",
        "summary": "Update NLP course readings with modern Transformer architectures and practical applications.",
    },
    {
        "id": 6,
        "category": "Labs & TA Support",
        "feedback": "The teaching assistants in the Thursday lab section were very knowledgeable and patient, but the lab machines frequently lacked required GPU libraries.",
        "summary": "Praise Thursday TAs for support; ensure all lab machines have pre-installed GPU libraries.",
    },
    {
        "id": 7,
        "category": "Exams & Alignment",
        "feedback": "The final exam included proof-heavy questions that were never practiced in homework or quizzes. It felt misaligned with the programming-heavy syllabus.",
        "summary": "Align exam content closely with homework and quiz emphasis on practical programming.",
    },
    {
        "id": 8,
        "category": "Office Hours",
        "feedback": "Office hours queue system was always overwhelmed before project deadlines. Having virtual queue appointments or additional slots would prevent 2-hour waits.",
        "summary": "Implement virtual queue appointments and expand office hour slots before major deadlines.",
    },
]


def get_synthetic_feedback_dataset() -> List[Dict[str, Any]]:
    """Return the curated synthetic student feedback dataset."""
    return [dict(item) for item in SYNTHETIC_FEEDBACK_DATA]


def save_feedback_dataset(filepath: str = "data/feedback_dataset.json") -> str:
    """Save the synthetic feedback dataset to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SYNTHETIC_FEEDBACK_DATA, indent=2))
    return str(path)


def load_feedback_dataset(filepath: str = "data/feedback_dataset.json") -> List[Dict[str, Any]]:
    """Load feedback dataset from disk or fallback to default."""
    path = Path(filepath)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return get_synthetic_feedback_dataset()


def milestone_1() -> Dict[str, Any]:
    """Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs."""
    dataset = get_synthetic_feedback_dataset()
    saved_path = save_feedback_dataset()
    categories = sorted(list({item["category"] for item in dataset}))
    return {
        "status": "success",
        "dataset": dataset,
        "count": len(dataset),
        "categories": categories,
        "saved_path": saved_path,
    }


# ==============================================================================
# Milestone 2: LoRA Adapter Layer, Injection, and Training
# ==============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer wrapping a standard PyTorch Linear layer.

    W_new = W0 + (lora_alpha / r) * (B @ A)
    where B is initialized to 0 and A is initialized with Kaiming uniform.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged or self.r == 0:
            return base_out
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Merge LoRA weights directly into the base layer."""
        if not self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from the base layer."""
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False


def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
) -> Tuple[nn.Module, List[str]]:
    """Inject LoRALinear layers into target linear modules of a model."""
    injected_modules = []

    # Freeze all existing parameters first
    for param in model.parameters():
        param.requires_grad = False

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                should_inject = False
                if target_modules is None:
                    should_inject = True
                else:
                    if any(target in full_name for target in target_modules):
                        should_inject = True

                if should_inject:
                    lora_layer = LoRALinear(
                        base_layer=child,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, name, lora_layer)
                    injected_modules.append(full_name)
            else:
                _replace(child, full_name)

    _replace(model)
    return model, injected_modules


def get_lora_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """Compute trainable vs total parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": round(trainable_percent, 4),
    }


def train_lora_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Optional[nn.Module] = None,
) -> float:
    """Run one optimization step updating only LoRA parameters."""
    model.train()
    optimizer.zero_grad()
    criterion = loss_fn or nn.MSELoss()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def milestone_2(r: int = 4, lora_alpha: float = 8.0, epochs: int = 5) -> Dict[str, Any]:
    """Milestone 2: Implement LoRA adapter injection and training."""
    # Construct a sample model representative of LLM projection heads
    torch.manual_seed(42)
    base_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )
    
    # Store initial weights to prove base weights remain untouched
    orig_w0 = base_model[0].weight.clone()
    
    # Inject LoRA
    adapted_model, injected = inject_lora(base_model, r=r, lora_alpha=lora_alpha)
    param_stats = get_lora_trainable_parameters(adapted_model)
    
    # Verify only LoRA parameters are trainable
    trainable_names = [name for name, p in adapted_model.named_parameters() if p.requires_grad]
    assert all("lora_" in name for name in trainable_names), "Only LoRA parameters should be trainable!"

    # Train LoRA on synthetic inputs
    optimizer = torch.optim.AdamW([p for p in adapted_model.parameters() if p.requires_grad], lr=0.05)
    synthetic_x = torch.randn(16, 64)
    synthetic_target = torch.randn(16, 32)
    
    losses = []
    for _ in range(epochs):
        loss_val = train_lora_step(adapted_model, optimizer, synthetic_x, synthetic_target)
        losses.append(round(loss_val, 5))
        
    # Check that base model weight remains unchanged during LoRA training
    assert torch.equal(adapted_model[0].base_layer.weight, orig_w0), "Base weights were altered during training!"

    return {
        "status": "success",
        "injected_modules": injected,
        "trainable_params": param_stats["trainable_params"],
        "all_params": param_stats["all_params"],
        "trainable_percent": param_stats["trainable_percent"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_history": losses,
    }


# --- Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
# Hint: Use Streamlit to build an intuitive web interface for interaction with your model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate the quality of generated summaries against original feedback.
# Hint: Implement methods to compare the generated summaries against your original feedback for quality assessment.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Utilize the lfllm-lflora technique to merge the trained adapter weights into the base model for deployment.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
