# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8qt38zwjps/badge.svg)](https://ask.thili.ai/b/8qt38zwjps)

This project develops a Student Feedback Summarizer, an EdTech tool designed for course instructors. It leverages the LoRA fine-tuning technique to adapt a pre-trained language model, enabling it to generate concise summaries from student comments. The goal is to streamline the review process, allowing instructors to rapidly identify key themes and improve their courses.

## What you are building

**Goal.** You will build a tool that fine-tunes a language model using LoRA to summarize student feedback, helping course instructors quickly understand comments for curriculum improvement.

**Done when.** The project is complete when you have a Streamlit application that can upload student feedback, fine-tune an LLM with LoRA to generate relevant summaries, and merge the adapter for efficient inference, demonstrating the effectiveness of the fine-tuning technique.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**  
      Hint: Focus on creating a small, well-structured dataset of input-output pairs.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the lfllm-lflora technique to inject and train the LoRA adapter on your chosen LLM.
- [ ] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**  
      Hint: Use Streamlit to build an intuitive web interface for interaction with your model.
- [ ] **4. Evaluate the quality of generated summaries against original feedback.**  
      Hint: Implement methods to compare the generated summaries against your original feedback for quality assessment.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Utilize the lfllm-lflora technique to merge the trained adapter weights into the base model for deployment.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the evaluation process by incorporating additional quantitative metrics or a user feedback mechanism within the Streamlit UI.
- Expand the Streamlit UI to include features like side-by-side comparison of generated summaries or the ability to edit summaries before saving.
- Experiment with different synthetic data generation strategies to improve the diversity and quality of the training dataset.
