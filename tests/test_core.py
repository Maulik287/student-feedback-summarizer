"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs.
def test_milestone_1():
    from app.core import milestone_1, get_synthetic_feedback_dataset, load_feedback_dataset
    
    result = milestone_1()
    assert result["status"] == "success"
    assert result["count"] >= 5
    assert len(result["dataset"]) == result["count"]
    
    for item in result["dataset"]:
        assert "feedback" in item and len(item["feedback"]) > 10
        assert "summary" in item and len(item["summary"]) > 5
        assert "category" in item
        
    loaded = load_feedback_dataset(result["saved_path"])
    assert len(loaded) == result["count"]



# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2():
    from app.core import milestone_2, LoRALinear, inject_lora, get_lora_trainable_parameters
    import torch
    import torch.nn as nn
    
    # 1. Test standalone LoRALinear layer
    linear = nn.Linear(32, 16)
    lora = LoRALinear(linear, r=4, lora_alpha=8.0)
    assert lora.base_layer.weight.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True
    
    x = torch.randn(4, 32)
    # Since lora_B is initialized to 0, initial output must equal base_layer output
    assert torch.allclose(lora(x), linear(x), atol=1e-6)
    
    # 2. Test milestone_2 end-to-end injection and training
    res = milestone_2(r=4, lora_alpha=8.0, epochs=5)
    assert res["status"] == "success"
    assert len(res["injected_modules"]) >= 2
    assert res["trainable_params"] > 0
    assert res["trainable_percent"] < 100.0  # Fraction of parameters trained
    assert res["final_loss"] < res["initial_loss"]  # Loss decreases



# Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
def test_milestone_3():
    from app.core import (
        milestone_3,
        summarize_feedback_rule_based,
        parse_uploaded_feedback,
    )

    res = milestone_3()
    assert res["status"] == "success"
    assert res["ui_ready"] is True
    assert len(res["sample_summary"]) > 5

    # Test parser with CSV and JSON formats
    csv_sample = "category,feedback\nLecture,Too fast\nLab,Helpful"
    parsed_csv = parse_uploaded_feedback(csv_sample, "data.csv")
    assert len(parsed_csv) == 2

    json_sample = '[{"id": 1, "feedback": "Great lecture slides", "summary": "Keep slides"}]'
    parsed_json = parse_uploaded_feedback(json_sample, "data.json")
    assert len(parsed_json) == 1
    assert parsed_json[0]["feedback"] == "Great lecture slides"


# Milestone 4: Evaluate the quality of generated summaries against original feedback.
def test_milestone_4():
    from app.core import (
        milestone_4,
        compute_token_f1,
        compute_rouge_l,
        compute_compression_ratio,
        evaluate_summary,
    )

    res = milestone_4()
    assert res["status"] == "success"
    assert res["total_evaluated"] >= 5
    assert res["average_compression_ratio"] < 1.0  # Summaries must be concise
    assert res["average_token_f1"] > 0.0

    # Unit test metrics
    ref = "Slow down lecture pacing on dynamic programming"
    cand = "Slow down lecture pace on dynamic programming"
    f1 = compute_token_f1(ref, cand)
    assert f1["f1"] > 0.6
    rouge = compute_rouge_l(ref, cand)
    assert rouge > 0.6

    comp = compute_compression_ratio("A very long student feedback paragraph with many words", "Short summary")
    assert comp < 0.5


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5():
    from app.core import (
        milestone_5,
        LoRALinear,
        merge_lora_into_linear,
        merge_lora_model,
    )
    import torch
    import torch.nn as nn

    res = milestone_5()
    assert res["status"] == "success"
    assert res["is_equivalent"] is True
    assert res["inference_ready"] is True
    assert res["max_diff_merged_adapter"] < 1e-5
    assert res["max_diff_clean_linear"] < 1e-5

    # Test merge_lora_model on a multi-layer network
    model = nn.Sequential(
        LoRALinear(nn.Linear(8, 8), r=2),
        nn.ReLU(),
        LoRALinear(nn.Linear(8, 4), r=2),
    )
    x = torch.randn(2, 8)
    out_before = model(x).clone()
    merged = merge_lora_model(model)
    
    # Ensure all LoRALinear layers were converted to pure nn.Linear
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    out_after = merged(x)
    assert torch.allclose(out_before, out_after, atol=1e-5)


