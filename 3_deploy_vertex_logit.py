# %%

# deploy_vertex_sklearn.py
from google.cloud import aiplatform
import csv
import sklearn, sys

PROJECT_ID = "fifth-sprite-475713-q9"
REGION = "us-central1"
ARTIFACT_URI = (
    "gs://fifth-sprite-475713-q9-bucket0/sklogit_pd_1m"  # folder z model.joblib
)
MODEL_DISPLAY_NAME = "sklogit-pd-1m"
ENDPOINT_DISPLAY_NAME = "sklogit-pd-1m-endpoint"

# Prebuilt scikit-learn prediction container (CPU)
# (dobierz regionowy prefix: us-/europe-/asia-)
ver = sklearn.__version__  # np. '1.5.1'
major_minor = ".".join(ver.split(".")[:2])  # '1.5'
image = f"us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.{major_minor.replace('.', '-')}:latest"
print("Use image:", image)
# SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-7:latest"
SERVING_IMAGE = image
# %%

# def main():
aiplatform.init(project=PROJECT_ID, location=REGION)

print("📦 Uploading model to Vertex AI…")
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=ARTIFACT_URI,
    serving_container_image_uri=SERVING_IMAGE,
)

# %%

print("🛰️  Creating (or using) endpoint and deploying…")
endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    traffic_split={"0": 100},
)
# %%

print("✅ Deployed!")
print("Model resource name:", model.resource_name)
print("Endpoint resource name:", endpoint.resource_name)

with open("resource_name.csv", mode="w", newline=None, encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([model.resource_name])
with open("endpoint_name.csv", mode="w", newline=None, encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([endpoint.resource_name])

# if __name__ == "__main__":
#     main()

# %%
