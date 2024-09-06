# This curl caller code calls the API executed by 'static_router_6_15.py'
# 
# Config 3: Route = 6| Utterances per route = 15
# Routes: physics, biology, history, philosophy, chemistry, computer_science
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
    "What is the periodic table?",  # Exact match for chemistry route
    "What is an algorithm?",  # Exact match for computer science route

    # Partial match prompts 'partial_match'
    "How does gravity affect objects?",  # Partial match for physics route
    "Explain the structure and function of DNA in cells.",  # Partial match for biology route
    "Tell me about an important historical figure.",  # Partial match for history route
    "What do philosophers think about reality?",  # Partial match for philosophy route
    "Describe the basic components of a chemical reaction.",  # Partial match for chemistry route
    "How does programming work?",  # Partial match for computer science route

    # Unrelated prompts (to test 'None' or 'no_match' classification)
    "What is the best way to cook pasta?",  # Unrelated
    "How do you play the piano?",  # Unrelated
    "What is the healthiest diet?",  # Unrelated
    "How do you repair a car engine?",  # Unrelated
    "What are the benefits of meditation?",  # Unrelated
    "Can you teach me how to paint?",  # Unrelated
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