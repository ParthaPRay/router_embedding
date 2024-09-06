# This code implements a static router by using semantic meaning from computed from cosine similarity to propagate the user's prompt to best found route
# Run this code first before 'curl_caller_2_5.py' code. It loads the whole semanctic static router on Ollama server that runs on fastAPI web server
# Aim of this code: To help find optimal threshold value for the routes, especially 'None' when no route is selected for a given embedded model

# Configuration 1: 
# Route = 2 | Utterances per route = 5
# Routes: physics, biology

#  Test Prompts: Exact match prompts (one for each route) + Partial match prompts (one for each route)+ 
#                Unrelated prompts (same number of route to test 'None' route classification)

#  Test this code by each of the embedding model
#
# Partha Pratim Ray
# 3 September, 2024


from fastapi import FastAPI, Request
import numpy as np
import requests
import threading
import psutil
import time
import csv
import os
import json
from pydantic import BaseModel
from queue import Queue
from statistics import mean

app = FastAPI()

# Define the embedding model and test mode
embed_model = "all-minilm:22m"
OLLAMA_API_URL = "http://localhost:11434/api/embed"

# CSV file setup
csv_file = 'static_embed_fastapi_logs.csv'
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
        name="physics",
        utterances=[
            "What is Newton's first law of motion?",
            "Can you explain the theory of relativity?",
            "What is quantum mechanics?",
            "How does gravity affect time?",
            "What is the speed of light in a vacuum?"
        ],
    ),
    Route(
        name="biology",
        utterances=[
            "What is the process of photosynthesis?",
            "Can you explain the structure of DNA?",
            "What is cellular respiration?",
            "How do ecosystems maintain balance?",
            "What is the role of mitochondria in cells?"
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
        
        if best_route is None:
            print("No matching route found.")
            route_name = None
        else:
            print(f"Selected Route: {best_route.name} with similarity: {similarity}")
            route_name = best_route.name

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
        similarity = round(similarity, 2) if similarity is not None else None

        # Calculate total duration and network latency
        total_duration = embed_metrics.get('total_duration', 0)
        load_duration = embed_metrics.get('load_duration', 0)
        prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
        
        # Network latency: time spent in network communication for the embedding request
        network_latency = total_duration - load_duration

        # Total response time: time from receiving the request to sending the response
        total_response_time = time.time() - start_time

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, embed_model, prompt, route_name, similarity, 'cosine', prompt_embedding,
            total_duration, load_duration, prompt_eval_count, avg_cpu_usage, 
            avg_memory_usage, network_latency, total_response_time
        ]

        # Put the log message into the CSV queue
        csv_queue.put(log_message_csv)

        return {
            "status": "success" if best_route else "no_match",
            "route_selected": route_name,
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

    return best_route if best_similarity > 0 else None, best_similarity if best_similarity > 0 else None

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
