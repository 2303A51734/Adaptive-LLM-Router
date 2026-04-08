import random
from typing import Tuple, Dict, Any
from .models import Observation, Action

PROMPT_DATASET = {
    "easy": [
        {"id": "e1", "preview": "Translate 'Hello, how are you?' to Spanish.", "tokens": 15, "complex": "low", "is_code": False},
        {"id": "e2", "preview": "What is the capital of Australia?", "tokens": 12, "complex": "low", "is_code": False},
        {"id": "e3", "preview": "Write a python print statement for Hello World.", "tokens": 10, "complex": "low", "is_code": True},
    ],
    "medium": [
        {"id": "m1", "preview": "Summarize this 4-paragraph email thread.", "tokens": 350, "complex": "moderate", "is_code": False},
        {"id": "m2", "preview": "Write a Python regex pattern to extract all valid IPv4 addresses.", "tokens": 120, "complex": "moderate", "is_code": True},
        {"id": "m3", "preview": "Draft a polite decline letter to a job candidate.", "tokens": 200, "complex": "moderate", "is_code": False},
    ],
    "hard": [
        {"id": "h1", "preview": "Analyze this 20-page financial 10-K report and extract the EBITDA.", "tokens": 4500, "complex": "high", "is_code": False},
        {"id": "h2", "preview": "Refactor this legacy C++ multi-threading codebase to use Rust concurrency.", "tokens": 4000, "complex": "high", "is_code": True},
        {"id": "h3", "preview": "Write a complete, secure Next.js authentication system.", "tokens": 2500, "complex": "high", "is_code": True},
    ]
}

class AdaptiveModelRoutingEnv:
    def __init__(self, target_difficulty="easy", max_steps=10):
        self.target_difficulty = target_difficulty 
        self.max_steps = max_steps
        self.current_step = 0
        self.history = []
        self.total_reward = 0.0
        self.current_state = None
        self.current_true_difficulty = None

    def _generate_task(self) -> Observation:
        if random.random() < 0.8:
            self.current_true_difficulty = self.target_difficulty
        else:
            self.current_true_difficulty = random.choice(["easy", "medium", "hard"])
            
        task_data = random.choice(PROMPT_DATASET[self.current_true_difficulty])
        system_load = random.choice(["low", "high"]) 
        
        return Observation(
            task_id=task_data["id"],
            prompt_preview=task_data["preview"],
            estimated_tokens=task_data["tokens"],
            complexity=task_data["complex"],
            is_code=task_data["is_code"],
            system_load=system_load,
            previous_actions=self.history[-3:]
        )

    def reset(self) -> Observation:
        self.current_step = 0
        self.history.clear()
        self.total_reward = 0.0
        self.current_state = self._generate_task()
        return self.current_state

    def state(self) -> Observation:
        return self.current_state

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        choice = action.model_choice
        self.history.append(choice)
        
        costs = {"use_small_model": 1, "use_medium_model": 3, "use_large_model": 6}
        cost = costs[choice]
        
        base_acc = {
            "use_small_model": {"easy": 0.9, "medium": 0.4, "hard": 0.05},
            "use_medium_model": {"easy": 0.98, "medium": 0.9, "hard": 0.4},
            "use_large_model": {"easy": 1.0, "medium": 0.98, "hard": 0.95}
        }
        
        accuracy = base_acc[choice][self.current_true_difficulty]
        if self.current_state.is_code and choice == "use_small_model":
            accuracy -= 0.2  
            
        accuracy = max(0.0, accuracy)
        
        latency_penalty = 0.0
        if self.current_state.system_load == "high":
            if choice == "use_large_model":
                latency_penalty = 4.0 
            elif choice == "use_medium_model":
                latency_penalty = 1.0
                
        reward = (accuracy * 10.0) - cost - latency_penalty
        
        self.total_reward += reward
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            "chosen": choice, 
            "cost": cost, 
            "latency_penalty": latency_penalty, 
            "accuracy": accuracy, 
            "reward_gained": round(reward, 2)
        }
        
        if not done:
            self.current_state = self._generate_task()
            
        return self.current_state, round(reward, 2), done, info

class BaseGrader:
    def __init__(self, target_diff):
        self.target_diff = target_diff
        
    def get_env(self): 
        return AdaptiveModelRoutingEnv(target_difficulty=self.target_diff, max_steps=10)
        
    def evaluate(self, env: AdaptiveModelRoutingEnv) -> float:
        # 1. Calculate the raw score
        # Using 90.0 as the denominator based on your code
        raw_score = env.total_reward / 90.0
        
        # 2. Strict Clamping (THIS IS THE FIX)
        # We use 0.01 and 0.99 to ensure we NEVER hit 0.0 or 1.0
        clamped_score = max(0.01, min(0.99, raw_score))
        
        return round(float(clamped_score), 4)


TASKS = {
    "task_easy_routing": BaseGrader("easy"),
    "task_medium_routing": BaseGrader("medium"),
    "task_hard_routing": BaseGrader("hard")
}