# This code calls the API URL by using threading almost concurrently 'POST' the prompts to the web server via 'curl'
# The prompts however are processed by the embedding model in sequential manner
# Partha Pratim Ray
# 29/08/2024


import threading
import requests
import json
from queue import Queue

# List of prompts
prompts = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest ocean on Earth?",
    "Why is the sky blue?",
    "What is photosynthesis?",
    "How do magnets work?",
    "Tell me a short story about a brave knight",
    "Write a short story about a trip to the moon",
    "Tell a short story about a lost treasure",
    "What is your favorite color?"
]

# API endpoint
api_url = "http://localhost:5000/process_prompt"

# Queue to store results
result_queue = Queue()

# Function to send a POST request to the API
def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps({"prompt": prompt}))
        result = response.json()
        result_queue.put((prompt, result))
    except Exception as e:
        result_queue.put((prompt, f"Error: {str(e)}"))

# Function to handle multiple prompts concurrently
def send_prompts_concurrently(prompts):
    threads = []
    for prompt in prompts:
        thread = threading.Thread(target=send_prompt, args=(prompt,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print results
    while not result_queue.empty():
        prompt, response = result_queue.get()
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

# Main function to execute the script
if __name__ == "__main__":
    send_prompts_concurrently(prompts)