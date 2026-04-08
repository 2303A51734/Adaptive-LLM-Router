import requests
from models import Action, Observation

class RoutingEnvClient:
    """
    A standard client to connect to the OpenEnv FastAPI container.
    Allows remote RL agents to train on this environment.
    """
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def reset(self) -> Observation:
        response = requests.post(f"{self.url}/reset")
        response.raise_for_status()
        return Observation(**response.json())

    def state(self) -> Observation:
        response = requests.get(f"{self.url}/state")
        response.raise_for_status()
        return Observation(**response.json())

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        payload = action.model_dump()
        response = requests.post(f"{self.url}/step", json=payload)
        response.raise_for_status()
        
        data = response.json()
        obs = Observation(**data["observation"])
        return obs, data["reward"], data["done"], data["info"]