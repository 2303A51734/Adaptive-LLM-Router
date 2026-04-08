import sys
import os
from fastapi import FastAPI
import gradio as gr
import uvicorn

# Relative imports for package structure
from .models import Action
from .env import AdaptiveModelRoutingEnv

app = FastAPI(title="LLM Routing OpenEnv API")
environment = AdaptiveModelRoutingEnv(target_difficulty="medium")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "OpenEnv LLM Router is running."}

@app.post("/reset")
def reset_env():
    return environment.reset().model_dump()

@app.get("/state")
def get_state():
    return environment.state().model_dump()

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = environment.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

def play_step(choice):
    action = Action(model_choice=choice)
    obs, reward, done, info = environment.step(action)
    status = f"**Reward Gained:** {reward}\n**Cost:** {info['cost']}\n**Latency Penalty:** {info['latency_penalty']}\n**Accuracy:** {info['accuracy']}"
    if done: status += "\n\n**Episode Complete!** Click Reset to play again."
    next_prompt = f"**Next Prompt Preview:** {obs.prompt_preview}\n**Complexity:** {obs.complexity} | **Tokens:** {obs.estimated_tokens} | **Server Load:** {obs.system_load} | **Is Code:** {obs.is_code}"
    return next_prompt, status

def reset_ui():
    obs = environment.reset()
    start_prompt = f"**Prompt Preview:** {obs.prompt_preview}\n**Complexity:** {obs.complexity} | **Tokens:** {obs.estimated_tokens} | **Server Load:** {obs.system_load} | **Is Code:** {obs.is_code}"
    return start_prompt, "Game Reset. Choose a model below!"

with gr.Blocks(title="AI Model Router") as demo:
    gr.Markdown("# 🚦 Adaptive AI Model Router")
    with gr.Row():
        state_box = gr.Markdown(value="Click **Reset Environment** to start.")
        result_box = gr.Markdown(value="")
    with gr.Row():
        btn_small = gr.Button("Route to Small Model (Cost: 1)")
        btn_medium = gr.Button("Route to Medium Model (Cost: 3)")
        btn_large = gr.Button("Route to Large Model (Cost: 6)")
    btn_reset = gr.Button("Reset Environment", variant="primary")
    
    btn_small.click(fn=lambda: play_step("use_small_model"), outputs=[state_box, result_box])
    btn_medium.click(fn=lambda: play_step("use_medium_model"), outputs=[state_box, result_box])
    btn_large.click(fn=lambda: play_step("use_large_model"), outputs=[state_box, result_box])
    btn_reset.click(fn=reset_ui, outputs=[state_box, result_box])

app = gr.mount_gradio_app(app, demo, path="/ui")

# Changed from start() to main() to match validation requirements
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
