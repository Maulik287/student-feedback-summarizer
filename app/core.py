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


# ==============================================================================
# Milestone 3: Summarization Logic and Streamlit Helpers
# ==============================================================================

def summarize_feedback_rule_based(feedback: str) -> str:
    """Generate an actionable summary from student feedback using keyword pattern synthesis."""
    text_lower = feedback.lower()
    actions = []
    
    if any(k in text_lower for k in ["fast", "pace", "rush", "slow"]):
        actions.append("Adjust lecture pace and allocate more time to challenging topics")
    if any(k in text_lower for k in ["homework", "assignment", "workload", "long"]):
        actions.append("Clarify assignment specifications and balance student workload")
    if any(k in text_lower for k in ["grading", "rubric", "feedback", "late"]):
        actions.append("Provide explicit grading rubrics and expedite assignment return times")
    if any(k in text_lower for k in ["textbook", "material", "outdated", "slides"]):
        actions.append("Refresh course readings with modern, real-world examples")
    if any(k in text_lower for k in ["ta", "office hours", "queue", "lab"]):
        actions.append("Enhance TA training and optimize office hours / lab scheduling")
    if any(k in text_lower for k in ["exam", "test", "midterm", "final", "proof"]):
        actions.append("Align examinations closer with homework exercises and quizzes")
    if any(k in text_lower for k in ["live code", "demo", "quiz", "interactive", "great", "love"]):
        actions.append("Maintain interactive live coding demos and active learning exercises")
        
    if not actions:
        first_sentence = feedback.split(".")[0].strip()
        return f"Review student feedback: {first_sentence}."
        
    return "; ".join(actions) + "."


def summarize_feedback(feedback: str, use_llm: bool = False) -> str:
    """Summarize student feedback using LLM if available, otherwise rule-based."""
    if use_llm:
        try:
            from app.llm import chat
            prompt = (
                "You are an assistant for course instructors. Summarize the following student feedback "
                "into 1-2 concise, actionable recommendations for course improvement:\n\n"
                f"Student Feedback: {feedback}\n\n"
                "Summary:"
            )
            return chat([{"role": "user", "content": prompt}]).strip()
        except Exception:
            pass
    return summarize_feedback_rule_based(feedback)


def parse_uploaded_feedback(content: str, filename: str) -> List[Dict[str, Any]]:
    """Parse feedback items from uploaded CSV, JSON, or plain text."""
    items = []
    fname = filename.lower()
    if fname.endswith(".json"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for i, row in enumerate(data):
                    if isinstance(row, dict) and "feedback" in row:
                        items.append({
                            "id": row.get("id", i + 1),
                            "category": row.get("category", "General"),
                            "feedback": row["feedback"],
                            "summary": row.get("summary", ""),
                        })
        except Exception:
            pass
    elif fname.endswith(".csv"):
        lines = content.strip().splitlines()
        if lines:
            header = [h.strip().lower() for h in lines[0].split(",")]
            fb_idx = header.index("feedback") if "feedback" in header else 0
            for i, line in enumerate(lines[1:], start=1):
                cols = [c.strip().strip('"') for c in line.split(",")]
                if len(cols) > fb_idx:
                    items.append({
                        "id": i,
                        "category": "CSV Upload",
                        "feedback": cols[fb_idx],
                        "summary": "",
                    })
    else:
        # Plain text, one feedback item per non-empty line
        for i, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if line:
                items.append({
                    "id": i,
                    "category": "Text Entry",
                    "feedback": line,
                    "summary": "",
                })
    return items


def milestone_3() -> Dict[str, Any]:
    """Milestone 3: Develop Streamlit UI helpers and summarization pipeline."""
    sample_feedback = "The professor went way too fast on dynamic programming, but the slides were clear."
    summary = summarize_feedback(sample_feedback)
    parsed = parse_uploaded_feedback(sample_feedback, "sample.txt")
    return {
        "status": "success",
        "sample_feedback": sample_feedback,
        "sample_summary": summary,
        "parsed_items_count": len(parsed),
        "ui_ready": True,
    }


# ==============================================================================
# Milestone 4: Evaluation of Generated Summaries
# ==============================================================================

def compute_token_f1(reference: str, candidate: str) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1 score."""
    ref_tokens = [w.strip(".,!?:;\"'()[]{}").lower() for w in reference.split() if w.strip()]
    cand_tokens = [w.strip(".,!?:;\"'()[]{}").lower() for w in candidate.split() if w.strip()]
    
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    common = ref_set.intersection(cand_set)
    
    prec = len(common) / len(cand_set) if cand_set else 0.0
    rec = len(common) / len(ref_set) if ref_set else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute Longest Common Subsequence (LCS) ROUGE-L F1."""
    ref = reference.lower().split()
    cand = candidate.lower().split()
    m, n = len(ref), len(cand)
    if m == 0 or n == 0:
        return 0.0
        
    # Standard dynamic programming table for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == cand[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    return round((2 * prec * rec / (prec + rec)), 4) if (prec + rec) > 0 else 0.0


def compute_compression_ratio(original: str, summary: str) -> float:
    """Compute length ratio of summary to original text."""
    orig_len = max(1, len(original.split()))
    summ_len = len(summary.split())
    return round(summ_len / orig_len, 4)


def evaluate_summary(
    original_feedback: str,
    generated_summary: str,
    reference_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate quality of generated summary against feedback and optional reference."""
    compression = compute_compression_ratio(original_feedback, generated_summary)
    
    # Check actionability
    action_words = [
        "adjust", "clarify", "balance", "provide", "refresh", "enhance", "align",
        "maintain", "improve", "slow", "expand", "streamline", "ensure", "update",
    ]
    summary_words = generated_summary.lower().split()
    actionable = any(w.strip(".,!?:;") in action_words for w in summary_words)
    
    metrics: Dict[str, Any] = {
        "compression_ratio": compression,
        "is_concise": compression < 1.0,
        "is_actionable": actionable,
        "summary_length_words": len(summary_words),
    }
    
    if reference_summary:
        f1_scores = compute_token_f1(reference_summary, generated_summary)
        rouge_l = compute_rouge_l(reference_summary, generated_summary)
        metrics["token_precision"] = f1_scores["precision"]
        metrics["token_recall"] = f1_scores["recall"]
        metrics["token_f1"] = f1_scores["f1"]
        metrics["rouge_l"] = rouge_l
        
    return metrics


def milestone_4() -> Dict[str, Any]:
    """Milestone 4: Evaluate quality of generated summaries against original feedback."""
    dataset = get_synthetic_feedback_dataset()
    eval_results = []
    
    for item in dataset:
        generated = summarize_feedback_rule_based(item["feedback"])
        metrics = evaluate_summary(item["feedback"], generated, item["summary"])
        eval_results.append({
            "id": item["id"],
            "feedback": item["feedback"],
            "reference": item["summary"],
            "generated": generated,
            "metrics": metrics,
        })
        
    avg_f1 = sum(r["metrics"]["token_f1"] for r in eval_results) / len(eval_results)
    avg_rouge = sum(r["metrics"]["rouge_l"] for r in eval_results) / len(eval_results)
    avg_compression = sum(r["metrics"]["compression_ratio"] for r in eval_results) / len(eval_results)
    
    return {
        "status": "success",
        "total_evaluated": len(eval_results),
        "average_token_f1": round(avg_f1, 4),
        "average_rouge_l": round(avg_rouge, 4),
        "average_compression_ratio": round(avg_compression, 4),
        "eval_results": eval_results,
    }


# ==============================================================================
# Milestone 5: LoRA Adapter Merging for Inference
# ==============================================================================

def merge_lora_into_linear(lora_layer: LoRALinear) -> nn.Linear:
    """Fuse LoRA weights permanently into a standard nn.Linear layer."""
    # Ensure weights are merged into base layer
    if not lora_layer.merged and lora_layer.r > 0:
        lora_layer.merge()
        
    # Construct a new pure nn.Linear module
    merged_linear = nn.Linear(
        in_features=lora_layer.in_features,
        out_features=lora_layer.out_features,
        bias=lora_layer.base_layer.bias is not None,
    )
    merged_linear.weight.data.copy_(lora_layer.base_layer.weight.data)
    if lora_layer.base_layer.bias is not None:
        merged_linear.bias.data.copy_(lora_layer.base_layer.bias.data)
    return merged_linear


def merge_lora_model(model: nn.Module) -> nn.Module:
    """Recursively replace all LoRALinear layers in a model with merged nn.Linear layers."""
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            setattr(model, name, merge_lora_into_linear(child))
        else:
            merge_lora_model(child)
    return model


def milestone_5() -> Dict[str, Any]:
    """Milestone 5: Implement adapter merging for inference and verify equivalence."""
    torch.manual_seed(42)
    # 1. Base model
    base_linear = nn.Linear(16, 8)
    
    # 2. Inject LoRA
    lora_layer = LoRALinear(base_linear, r=4, lora_alpha=8.0)
    # Simulate trained adapter weights (non-zero B)
    lora_layer.lora_B.data.normal_(0.0, 0.05)
    
    # 3. Test input
    test_x = torch.randn(4, 16)
    
    # 4. Output with adapter active (before merge)
    out_before_merge = lora_layer(test_x).detach().clone()
    
    # 5. Merge adapter
    lora_layer.merge()
    out_after_merge = lora_layer(test_x).detach().clone()
    
    # Check numerical equivalence
    diff = torch.max(torch.abs(out_before_merge - out_after_merge)).item()
    assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5), f"Merged output differs: diff={diff}"
    
    # 6. Fuse into clean nn.Linear
    clean_linear = merge_lora_into_linear(lora_layer)
    out_clean = clean_linear(test_x).detach().clone()
    diff_clean = torch.max(torch.abs(out_before_merge - out_clean)).item()
    assert torch.allclose(out_before_merge, out_clean, atol=1e-5), f"Clean linear differs: diff={diff_clean}"
    
    return {
        "status": "success",
        "max_diff_merged_adapter": diff,
        "max_diff_clean_linear": diff_clean,
        "is_equivalent": True,
        "inference_ready": True,
    }

