
import os
import shutil
import logging

from tqdm import tqdm
from google.cloud import storage
from google.cloud import aiplatform
from fastapi import FastAPI, Request
from fastapi.logger import logger


from starlette.responses import JSONResponse
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import torch


app = FastAPI()

PROJECT_ID = "argolis-rafaelsanchez-ml-dev"
AIP_PROJECT_NUMBER=os.getenv("AIP_PROJECT_NUMBER", "989788194604")
AIP_PREDICT_ROUTE=os.getenv("AIP_PREDICT_ROUTE", "/predict")
AIP_HEALTH_ROUTE=os.getenv("AIP_HEALTH_ROUTE", "/health")
AIP_STORAGE_URI=os.getenv("AIP_STORAGE_URI", "gs://argolis-rafaelsanchez-llm-models/llama-2-chat-70b-gptq")
LOCAL_MODEL_DIR="llama-2-chat-70b-gptq/"



gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.INFO)

logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")



logger.info(AIP_HEALTH_ROUTE)
logger.info(AIP_PREDICT_ROUTE)

#os.mkdir(LOCAL_MODEL_DIR)

# aiplatform.init(project=PROJECT_ID)

# storage_client = storage.Client(AIP_PROJECT_NUMBER)
# bucket = storage_client.bucket(AIP_STORAGE_URI.split("/")[2])
# blobs = bucket.list_blobs(prefix=AIP_STORAGE_URI.split("/")[3])
# for blob in blobs:
#     logger.info(f"Free Disk: {shutil.disk_usage(__file__)[2]/1024/1024/1024}")
#     logger.info(blob.name.split("/")[-1])
#     if blob.name.split("/")[-1] != "":
#         filename = blob.name.split("/")[-1]
#         with open(LOCAL_MODEL_DIR+filename, "wb") as in_file:
#             with tqdm.wrapattr(in_file, "write", total=blob.size, miniters=1, desc="Downloading") as destination_file_name:
#                 storage_client.download_blob_to_file(blob, destination_file_name) 


LOCAL_MODEL_DIR = '../llama2-70b-chat-gptq'


logger.info(f"Loading model {LOCAL_MODEL_DIR}. This takes some time ...")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(LOCAL_MODEL_DIR,
        model_basename="model",
        inject_fused_attention=False, # Required for Llama 2 70B model at this time.
        use_safetensors=True,
        trust_remote_code=False,
        #device="cuda:0",
        use_triton=False,
        quantize_config=None,
        device_map="auto"
        )

logger.info(f"Loading model DONE")


    
@app.get(AIP_HEALTH_ROUTE, status_code=200)
def health():
    return dict(status="healthy")

@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request, status_code=200):
    body = await request.json()
    prompt = body["instances"]
    
    system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt_template=f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {prompt} [/INST]'''

    inputs = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    generated_ids = model.generate(inputs=inputs, temperature=0.7, max_new_tokens=254)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return JSONResponse({"predictions": response})




# curl -X GET \
#     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
#     "https://europe-west4-aiplatform.googleapis.com/v1/projects/989788194604/locations/europe-west4/endpoints/7034662200371314688/operations/6857616913921474560"