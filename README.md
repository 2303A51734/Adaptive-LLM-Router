---
license: mit
title: Adaptive-LLM-Router
sdk: docker
tags:
- openenv
---
---
# Adaptive AI Model Routing Agent

## Environment Overview & Motivation
In production AI environments, sending every user query to frontier models (like GPT-4 Opus or Claude 3.5 Sonnet) is prohibitively expensive. This OpenEnv simulates a critical real-world MLOps task: **LLM Routing**. The goal of the AI agent is to act as a router, classifying incoming prompts and routing them to Small, Medium, or Large models to maximize accuracy while minimizing compute cost.

## Action and Observation Space
* **Observation**: Typed via Pydantic (`models.py`). Contains `task_id`, `prompt_preview`, `estimated_tokens`, `complexity_category`, and `previous_actions`.
* **Action**: The agent chooses a discrete model: `use_small_model`, `use_medium_model`, or `use_large_model`.

## Tasks & Difficulty
1. `task_easy_routing`: Majority of tasks are low complexity. The agent must learn to favor small models to maximize cost-efficiency.
2. `task_medium_routing`: Mixed complexity. Tests dynamic switching.
3. `task_hard_routing`: High complexity tasks where cheap models fail. Agent must correctly spend money on large models.

## Setup Instructions
1. Ensure you have Docker installed.
2. Build the image:
   ```bash
   docker build -t openenv-router .