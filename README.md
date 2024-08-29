# router_embedding
This repo contains a code that uses embedding model to enroute the prompt to appropriate route


# Always run the code inside the virtual enviornment

# First start the FastAPI server at 5000 port

```bash
python3 router_dot_default.py
```

It should show below

```bash
INFO:     Started server process [7325]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```

# Curl the prompt to the server at 5000 port

Call the @ **/process_prompt** by below curl command

```bash
curl -X POST http://localhost:5000/process_prompt      -H "Content-Type: application/json"      -d '{"prompt": "What is your political opinion?"}'
```


OR

Run below **curl_caller.py** that shoots many prompts to the FastAPI server simultaneously, however the embedding model processes each prompt sequentially. 

```python
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
```

