""" 
    Deploy image with Uvicorn server containing FastAPI app and a Llama2-70B-chat model in Vertex AI
    The deployment uses a g2-standard-24 machine type with 2xL4 GPU
"""
    
from google.cloud.aiplatform import Model, Endpoint

DEPLOY_IMAGE="europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/llama2-70b-chat"
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [7080]

model = Model.upload(
    display_name="llama2-70B-chat", 
    description=f'llama2-70B-chat with Uvicorn and FastAPI',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
    location="europe-west4",
    upload_request_timeout=1800,
    sync=True,
    )

# Retrieve a Model on Vertex
model = Model(model.resource_name)

# Deploy model
endpoint = model.deploy(
    machine_type="g2-standard-24",
    accelerator_type="NVIDIA_L4",
    accelerator_count=2,
    traffic_split={"0": 100}, 
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    deploy_request_timeout=1800,
    sync=True,
)
endpoint.wait()