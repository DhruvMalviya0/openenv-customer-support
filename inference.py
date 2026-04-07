import os
import json
import requests
from openai import OpenAI

# ==============================================================================
# REQUIRED CONFIGURATION (Strict adherence to OpenEnv Phase 2 requirements)
# ==============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Initialize OpenAI Client configured via the exact variables above
client = OpenAI(
    base_url=API_BASE_URL,
    # Use HF_TOKEN if provided by the grader, otherwise fallback to standard OpenAI key
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key")
)

# Local environment URL (matches your FastAPI setup)
ENV_URL = "http://localhost:7860"

def run_episode(task_id: str, task_description: str, action_schema: dict):
    # Log the exact required START tag
    print("START")
    print(f"Initializing task: {task_id}")
    
    # Reset Environment
    reset_res = requests.post(f"{ENV_URL}/reset")
    if reset_res.status_code != 200:
        print(f"Failed to reset environment: {reset_res.text}")
        print("END")
        return
        
    observation = reset_res.json()
    done = False
    step_count = 0
    max_steps = 10
    
    while not done and step_count < max_steps:
        step_count += 1
        
        # Log the exact required STEP tag
        print("STEP")
        print(f"Observation: {observation}")
        
        prompt = f"""
        You are an AI support agent. Your goal is: {task_description}
        
        Current Observation:
        {json.dumps(observation, indent=2)}
        
        Allowed Actions Schema:
        {json.dumps(action_schema, indent=2)}
        
        Output ONLY a valid JSON object representing your next action.
        """
        
        try:
            # All LLM calls use the OpenAI client configured via the required variables
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            action_payload = json.loads(response.choices[0].message.content)
            print(f"Agent decided action: {action_payload}")
            
        except Exception as e:
            print(f"LLM Error or JSON parsing failed: {e}")
            print("END")
            return
            
        # Send action to environment
        step_res = requests.post(f"{ENV_URL}/step", json=action_payload)
        if step_res.status_code != 200:
            print(f"Failed to step environment: {step_res.text}")
            print("END")
            return
            
        step_data = step_res.json()
        observation = step_data["observation"]
        done = step_data["done"]

    # Get Final Grade
    grade_res = requests.get(f"{ENV_URL}/grader", params={"task_id": task_id})
    score = grade_res.json().get("score", 0.0) if grade_res.status_code == 200 else 0.0
    
    # Log the exact required END tag
    print("END")
    print(f"Task {task_id} complete. Steps: {step_count}. Final Score: {score}\n")

if __name__ == "__main__":
    try:
        # Fetch tasks from the environment
        tasks_res = requests.get(f"{ENV_URL}/tasks")
        if tasks_res.status_code == 200:
            tasks = tasks_res.json()
            for task_id, task_info in tasks.items():
                run_episode(task_id, task_info["description"], task_info["action_schema"])
        else:
            print("Failed to fetch tasks from environment.")
    except Exception as e:
        print(f"Failed to connect to environment. Ensure server is running. Error: {e}")
