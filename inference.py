import os
import json
from openai import OpenAI

# Change the dots to 'server.' because these files are inside the server folder
from server.env import TASKS
from server.models import Action, Observation


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "mock_token")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def get_agent_action(obs: Observation) -> Action:
    # Safely handle offline testing / mocked keys
    if HF_TOKEN in ["mock_token", "dummy_token", "mock", ""]:
        return _smart_fallback(obs)
        
    prompt = (
        f"Route this task based on complexity, cost, and server load.\n"
        f"Complexity: {obs.complexity} | Tokens: {obs.estimated_tokens} | Is Code: {obs.is_code}\n"
        f"Server Load: {obs.system_load} (If load is high, avoid large models!)\n"
        f"Options: use_small_model, use_medium_model, use_large_model."
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            timeout=10.0,
            temperature=0.0
        )
        choice = response.choices[0].message.content.strip().lower()
        if choice not in ["use_small_model", "use_medium_model", "use_large_model"]:
            raise ValueError("Bad response")
    except Exception:
        choice = _smart_fallback(obs).model_choice
            
    return Action(model_choice=choice)

def _smart_fallback(obs: Observation) -> Action:
    if obs.complexity == "low":
        choice = "use_small_model" if not obs.is_code else "use_medium_model"
    elif obs.complexity == "moderate":
        if obs.system_load == "high":
            choice = "use_small_model" 
        else:
            choice = "use_medium_model"
    else: 
        if obs.system_load == "high" and not obs.is_code:
            choice = "use_medium_model" 
        else:
            choice = "use_large_model"
    return Action(model_choice=choice)

def run_evaluation():
    print("[START] inference.py initialized")
    for task_name, grader in TASKS.items():
        print(f"[START] task={task_name}")
        env = grader.get_env()
        obs = env.reset()
        done = False
        while not done:
            action = get_agent_action(obs)
            obs, reward, done, info = env.step(action)
            action_json = json.dumps({"model_choice": action.model_choice})
            print(f"[STEP] action={action_json} reward={reward:.2f} done={done} info={json.dumps(info)}")
        score = grader.evaluate(env)
        print(f"[END] task={task_name} score={score:.4f}")
    print("[END] all tasks completed")

if __name__ == "__main__":
    run_evaluation()