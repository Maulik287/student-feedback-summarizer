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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate the quality of generated summaries against original feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

