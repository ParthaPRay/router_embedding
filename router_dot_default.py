# This code routes a given prompt to the best possible route based on semantic similarity using FastAPI web server using Ollama API.
# This code uses dot product.
# This code includes a default route when no given route is selected.
# This code always return the best matching route, even if the similarity is low.
# Partha Pratim Ray
# 29/08/2024

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
        network_latency = total_duration - load_duration

        # Total response time: time from receiving the request to sending the response
        total_response_time = time.time() - start_time

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, embed_model, prompt, best_route.name, similarity, 'dot_product', prompt_embedding,
            total_duration, load_duration, prompt_eval_count, avg_cpu_usage, 
            avg_memory_usage, network_latency, total_response_time
        ]

        # Put the log message into the CSV queue
        csv_queue.put(log_message_csv)

        return {
            "status": "success",
            "route_selected": best_route.name,
            "similarity_score": similarity,
            "similarity_metric": "dot_product",
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

# Function to calculate dot product similarity
def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

# Function to find the best route based on the prompt
def find_best_route(prompt_embedding, routes):
    best_route = None
    best_similarity = -float('inf')  # Initialize to a very low value since we seek the maximum dot product

    for route in routes:
        for utterance in route.utterances:
            utterance_embedding, _ = get_embedding(utterance)
            similarity = dot_product(prompt_embedding, utterance_embedding)
            if similarity > best_similarity:  # Higher dot product indicates more similarity
                best_similarity = similarity
                best_route = route

    return best_route, best_similarity

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
