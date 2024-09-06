# This curl caller code calls the API executed by 'static_router_4_10.py'
# 
# Config 2: Route = 4| Utterances per route = 10
# Routes: physics, biology, history, philosophy
#  Test Prompts: Exact match prompts (one for each route) + Partial match prompts (one for each route)+ 
#                Unrelated prompts (same number of route to test 'None' route classification)
#  Test this code by each of the embedding model
#
# Partha Pratim Ray
# 3 September, 2024

import requests
import json

# List of test prompts
prompts = [
    # Exact match prompts 'exact_match'
    "What is Newton's first law of motion?",  # Exact match for physics route
    "What is the process of photosynthesis?",  # Exact match for biology route
    "Who was the first president of the United States?",  # Exact match for history route
    "What is existentialism?",  # Exact match for philosophy route

    # Partial match prompts 'partial_match'
    "Explain motion in simple physics terms.",  # Partial match for physics route
    "Describe how plants convert sunlight to energy.",  # Partial match for biology route
    "Talk about the major events of the American Revolution.",  # Partial match for history route
    "Discuss the meaning of free will.",  # Partial match for philosophy route

    # Unrelated prompts (to test 'None' or 'no_match' classification)
    "How does a computer process data?",  # Unrelated prompt to test 'None' classification
    "What are the benefits of a balanced diet?",  # Unrelated prompt to test 'None' classification
    "Can you explain blockchain technology?",  # Unrelated prompt to test 'None' classification
    "What is the capital city of France?"  # Unrelated prompt to test 'None' classification
]

# API endpoint
api_url = "http://localhost:5000/process_prompt"

# Function to send a POST request to the API
def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps({"prompt": prompt}))
        result = response.json()
        print(f"Prompt: {prompt}")
        print(f"Response: {result}\n")
    except Exception as e:
        print(f"Error for prompt '{prompt}': {str(e)}\n")

# Main function to execute the script
if __name__ == "__main__":
    for prompt in prompts:
        send_prompt(prompt)