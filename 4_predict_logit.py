# predict_vertex_sklearn.py
from google.cloud import aiplatform
import csv

# === CONFIGURATION ===
PROJECT_ID = "fifth-sprite-475713-q9"
REGION = "us-central1"

# wczytaj endpoint z poprzedniego deploya
with open("endpoint_name.csv") as f:
    ENDPOINT_NAME = f.read().strip()

# dane wejściowe – kolejność zgodna z feature_list.json
values = [0, 1_000_000, 500_000, 1000, 1000, 1000, 1000, 1000, 300_000]

# === INFERENCE ===
aiplatform.init(project=PROJECT_ID, location=REGION)

endpoint = aiplatform.Endpoint(ENDPOINT_NAME)

# Vertex AI oczekuje listy rekordów (możesz wysłać batch)
instances = [values]

response = endpoint.predict(instances=instances)

print("✅ Prediction response:")
print(response)
print("Predicted probabilities:", response.predictions)
print("Deployed model resource name:", response.deployed_model_id)
