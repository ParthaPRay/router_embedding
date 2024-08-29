# router_embedding
This repo contains a code that uses embedding model to enroute the prompt to appropriate route


# Always run the code inside the virtual enviornment

# The Basic Router Code

Below python code **router_cosine_default.py** saves various metrics into the csv file after performing cosine similarity between the user given prompt and the pre-existing route prompts. It finds best ruote and enroutes the prompt to the selected route. If no pre-defined route is selected, it propagates the prompt to the **default** route.

```python
from fastapi import FastAPI, Request
import numpy as np
import requests
import threading
import psutil
import time
import csv
import os
from pydantic import BaseModel
from queue import Queue
from statistics import mean

app = FastAPI()

# Define the embedding model and test mode
embed_model = "nomic-embed-text"
OLLAMA_API_URL = "http://localhost:11434/api/embed"

# CSV file setup
csv_file = 'embed_fastapi_logs.csv'
csv_headers = [
    'timestamp', 'embed_model', 'prompt', 'route_selected',
    'semantic_similarity_score', 'similarity_metric', 'vector', 'total_duration',
    'load_duration', 'prompt_eval_count', 'avg_cpu_usage_during',
    'memory_usage_mb', 'network_latency', 'total_response_time'
]

csv_queue = Queue()
cpu_usage_queue = Queue()
memory_usage_queue = Queue()
is_monitoring = False

def csv_writer():
    while True:
        log_message_csv = csv_queue.get()
        if log_message_csv is None:  # Exit signal
            break
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(csv_headers)
            writer.writerow(log_message_csv)

# Start the CSV writer thread
csv_thread = threading.Thread(target=csv_writer)
csv_thread.start()

class Prompt(BaseModel):
    prompt: str

# Define routes with sample utterances
class Route:
    def __init__(self, name, utterances):
        self.name = name
        self.utterances = utterances

routes = [
    Route(
        name="politics",
        utterances=[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions",
            "don't you just love the president",
            "don't you just hate the president",
            "they're going to destroy this country!",
            "they will save the country!",
        ],
    ),
    Route(
        name="chitchat",
        utterances=[
            "how's the weather today?",
            "how are things going?",
            "lovely weather today",
            "the weather is horrendous",
            "let's go to the chippy",
        ],
    ),
    # Add a default route
    Route(
        name="default",
        utterances=[
            "I'm not sure how to categorize this",
            "This is a general statement",
            "This could be about anything",
        ],
    ),
]

def monitor_resources():
    global is_monitoring
    process = psutil.Process()
    while is_monitoring:
        cpu_usage = psutil.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        cpu_usage_queue.put(cpu_usage)
        memory_usage_queue.put(memory_usage)
        time.sleep(0.01)  # Poll every 10ms

@app.post("/process_prompt")
async def process_prompt(request: Prompt):
    global is_monitoring
    start_time = time.time()
    data = request.dict()
    prompt = data['prompt']

    try:
        # Start resource monitoring
        is_monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Get embedding for the prompt
        prompt_embedding, embed_metrics = get_embedding(prompt)
        
        # Find the best route based on the prompt
        best_route, similarity = find_best_route(prompt_embedding, routes)
        print(f"Selected Route: {best_route.name} with similarity: {similarity}")

        # Stop resource monitoring
        is_monitoring = False
        monitor_thread.join()

        # Calculate resource usage statistics
        cpu_usages = []
        memory_usages = []
        while not cpu_usage_queue.empty() and not memory_usage_queue.empty():
            cpu_usages.append(cpu_usage_queue.get())
            memory_usages.append(memory_usage_queue.get())
        
        avg_cpu_usage = round(mean(cpu_usages), 2) if cpu_usages else 0
        avg_memory_usage = round(mean(memory_usages), 2) if memory_usages else 0
        similarity = round(similarity, 2)  # Round the similarity score

        # Calculate total duration and network latency
        total_duration = embed_metrics.get('total_duration', 0)
        load_duration = embed_metrics.get('load_duration', 0)
        prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
        
        # Network latency: time spent in network communication for the embedding request
        # This is the difference between total duration and load duration of the embedding process
        network_latency = total_duration - load_duration

        # Total response time: time from receiving the request to sending the response
        # This includes all processing time, not just the embedding request
        total_response_time = time.time() - start_time

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, embed_model, prompt, best_route.name, similarity, 'cosine', prompt_embedding,
            total_duration, load_duration, prompt_eval_count, avg_cpu_usage, 
            avg_memory_usage, network_latency, total_response_time
        ]

        # Put the log message into the CSV queue
        csv_queue.put(log_message_csv)

        return {
            "status": "success",
            "route_selected": best_route.name,
            "similarity_score": similarity,
            "similarity_metric": "cosine",
            "embedding": prompt_embedding,
            "avg_cpu_usage": avg_cpu_usage,
            "avg_memory_usage_mb": avg_memory_usage,
            "network_latency": network_latency,
            "total_response_time": total_response_time
        }
    except Exception as e:
        is_monitoring = False  # Ensure monitoring stops in case of an error
        return {"status": "error", "message": str(e)}

# Function to get embeddings from Ollama
def get_embedding(text, model=embed_model):
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": model, "input": text}
    )
    response_json = response.json()
    return response_json["embeddings"][0], response_json

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to find the best route based on the prompt
def find_best_route(prompt_embedding, routes):
    best_route = None
    best_similarity = -1

    for route in routes:
        for utterance in route.utterances:
            utterance_embedding, _ = get_embedding(utterance)
            similarity = cosine_similarity(prompt_embedding, utterance_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route

    return best_route, best_similarity

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```
    

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

