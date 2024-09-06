# This code implements a static router by using semantic meaning from computed from cosine similarity to propagate the user's prompt to best found route
# Run this code first before'curl_caller_8_20.py' code. It loads the whole semanctic static router on Ollama server that runs on fastAPI web server
# Aim of this code: To help find optimal threshold value for the routes, especially 'None' when no route is selected for a given embedded model

# Configuration 4: 
# Route = 8 | Utterances per route = 20
# Routes: physics, biology, history, philosophy, chemistry, computer_science, sculpture, dance

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
csv_file = 'static_embed_fastapi_logs_config2.csv'
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
            "Explain the concept of gravity.",
            "What is the theory of relativity?",
            "Define acceleration and velocity.",
            "What is the speed of light?",
            "Explain quantum mechanics.",
            "What is electromagnetic radiation?",
            "Define kinetic energy and potential energy.",
            "What are the laws of thermodynamics?",
            "How does friction affect motion?",
            "What is a black hole?",
            "Explain the concept of time dilation.",
            "What is the photoelectric effect?",
            "How do magnets work?",
            "What is string theory?",
            "What is a quark?",
            "Explain the concept of dark matter.",
            "What is a neutron star?",
            "What is the difference between mass and weight?",
            "How do waves propagate?"
        ],
    ),
    Route(
        name="biology",
        utterances=[
            "What is the process of photosynthesis?",
            "Explain the structure of DNA.",
            "What is cell division?",
            "Define the role of mitochondria.",
            "What is natural selection?",
            "Describe the process of evolution.",
            "What are the components of a cell?",
            "Explain the function of the nervous system.",
            "What is an ecosystem?",
            "How do plants reproduce?",
            "What is a genetic mutation?",
            "Describe the human immune system.",
            "What is homeostasis?",
            "Explain the function of enzymes.",
            "What is the role of RNA?",
            "What is a virus?",
            "Describe the process of fermentation.",
            "What is the difference between mitosis and meiosis?",
            "What is the role of chlorophyll?",
            "Explain the concept of biodiversity."
        ],
    ),
    Route(
        name="history",
        utterances=[
            "Who was the first president of the United States?",
            "What caused the fall of the Roman Empire?",
            "Describe the events of the French Revolution.",
            "What is the significance of the Magna Carta?",
            "Who was Alexander the Great?",
            "What started World War II?",
            "What was the Industrial Revolution?",
            "Who were the founding fathers of the USA?",
            "What was the Renaissance?",
            "Describe the ancient Egyptian civilization.",
            "What is the significance of the Great Wall of China?",
            "What was the Cold War?",
            "Who was Julius Caesar?",
            "What is the history of the Silk Road?",
            "What were the Crusades?",
            "What was the impact of the American Civil War?",
            "Describe the history of ancient Greece.",
            "Who was Napoleon Bonaparte?",
            "What led to the fall of the Berlin Wall?",
            "What is the significance of the Battle of Waterloo?"
        ],
    ),
    Route(
        name="philosophy",
        utterances=[
            "What is existentialism?",
            "Define the concept of free will.",
            "What is the meaning of life?",
            "Explain the philosophy of Stoicism.",
            "What is dualism?",
            "Who was Socrates?",
            "What is the philosophy of ethics?",
            "What is a moral dilemma?",
            "Explain the theory of knowledge.",
            "What is metaphysics?",
            "Describe the philosophy of utilitarianism.",
            "What is the nature of reality?",
            "What is the concept of the soul?",
            "What is the problem of evil?",
            "What is a logical fallacy?",
            "What is the philosophy of language?",
            "Explain the concept of phenomenology.",
            "What is the role of philosophy in science?",
            "What is the philosophy of mind?",
            "What is the debate on free will vs determinism?"
        ],
    ),
    Route(
        name="chemistry",
        utterances=[
            "What is the periodic table?",
            "Explain the process of chemical bonding.",
            "What is an atom?",
            "Define the law of conservation of mass.",
            "What is a chemical reaction?",
            "What are acids and bases?",
            "Explain the concept of molarity.",
            "What is a catalyst?",
            "Describe the structure of water molecules.",
            "What is a polymer?",
            "Explain the pH scale.",
            "What are hydrocarbons?",
            "What is the difference between organic and inorganic chemistry?",
            "What is an isotope?",
            "Describe the process of oxidation and reduction.",
            "What is a valence electron?",
            "Explain the concept of electronegativity.",
            "What is Avogadro's number?",
            "What is a chemical equilibrium?",
            "What are noble gases?"
        ],
    ),
    Route(
        name="computer_science",
        utterances=[
            "What is an algorithm?",
            "Define the concept of a programming language.",
            "What is machine learning?",
            "Explain the binary number system.",
            "What is artificial intelligence?",
            "What is the internet?",
            "Describe the function of a CPU.",
            "What is a database?",
            "Explain the concept of object-oriented programming.",
            "What is cybersecurity?",
            "What is a network protocol?",
            "What is blockchain technology?",
            "What is cloud computing?",
            "Define software development lifecycle.",
            "What is data mining?",
            "What is a neural network?",
            "Explain the concept of big data.",
            "What is an API?",
            "What is the difference between frontend and backend development?",
            "What is quantum computing?"
        ],
    ),
    Route(
        name="sculpture",
        utterances=[
            "What is sculpture?",
            "Describe the process of sculpting.",
            "What materials are used in sculpture?",
            "Who is a famous sculptor?",
            "What is the history of sculpture?",
            "What are the different types of sculpture?",
            "Explain the concept of abstract sculpture.",
            "What is a relief sculpture?",
            "What is the significance of Michelangelo's David?",
            "Describe the art of bronze casting.",
            "What is contemporary sculpture?",
            "How is marble used in sculpture?",
            "What is a statue?",
            "Explain the use of clay in sculpture.",
            "What is kinetic sculpture?",
            "What is a bust sculpture?",
            "Describe the technique of stone carving.",
            "What is modern sculpture?",
            "Who was Auguste Rodin?",
            "What is the purpose of public sculptures?"
        ],
    ),
    Route(
        name="dance",
        utterances=[
            "What is ballet?",
            "Describe the history of dance.",
            "What is contemporary dance?",
            "What is the significance of dance in culture?",
            "Who is a famous dancer?",
            "What are the different styles of dance?",
            "Explain the concept of modern dance.",
            "What is hip-hop dance?",
            "Describe the art of classical dance.",
            "What is tap dancing?",
            "What is the role of choreography?",
            "What is a dance studio?",
            "What is the significance of dance therapy?",
            "What is a dance recital?",
            "How does dance improve physical fitness?",
            "What is jazz dance?",
            "Describe the technique of ballet pointe work.",
            "What is the role of music in dance?",
            "What is salsa dancing?",
            "What is the history of folk dance?"
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
