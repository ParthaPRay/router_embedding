# This curl caller code calls the API executed by 'static_router_2_5.py'
# 
# Config 1: Route = 2 | Utterances per route = 5
# Routes: physics, biology
#  Test Prompts: Exact match prompts (one for each route) + Partial match prompts (one for each route)+ 
#                Unrelated prompts (same number of route to test 'None' route classification)
#  Test this code by each of the embedding model
#
# Partha Pratim Ray
# 3 September, 2024

import requests
import json
from queue import Queue

# List of test prompts
prompts = [
    # Exact match prompts 'exact_match'
    "What is Newton's first law of motion?",  # Exact match for physics route
    "What is the process of photosynthesis?",  # Exact match for biology route

    # Partial match prompts 'partial_match'
    "Can you explain motion in physics?",  # Partial match for physics route
    "Tell me about cell structures and their functions.",  # Partial match for biology route

    # Unrelated prompts (to test 'None' or 'no_match' classification)
    "How to prepare a delicious pasta recipe?",  # Completely unrelated prompt
    "What is the best way to meditate and relieve stress?"  # Completely unrelated prompt
]

# API endpoint
api_url = "http://localhost:5000/process_prompt"

# Function to send a POST request to the API
def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps({"prompt": prompt}))
        result = response.json()
        return (prompt, result)
    except Exception as e:
        return (prompt, f"Error: {str(e)}")

# Main function to execute the script
if __name__ == "__main__":
    # Loop through each prompt sequentially and send to the API
    for prompt in prompts:
        prompt_result, response = send_prompt(prompt)
        print(f"Prompt: {prompt_result}")
        print(f"Response: {response}\n")
