""" 
    Simple local inference of Llama 2-7B model
    Tested in Vertex AI Workbench Instances with 2xL4
    !pip install --no-cache-dir -U torch --extra-index-url https://download.pytorch.org/whl/nightly/cu121
    !pip install auto-gptq auto-gptq[triton] 
"""

import os
import shutil

from tqdm import tqdm
from fastapi import FastAPI, Request

from starlette.responses import JSONResponse
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import torch


app = FastAPI()

print(f"Is CUDA available: {torch.cuda.is_available()}")

print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

LOCAL_MODEL_DIR = '../llama2-70b-chat-gptq'

print(f"Loading model {LOCAL_MODEL_DIR}. This takes some time ...")

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

print(f"Loading model DONE")


prompt = "Que puedo ver en Murcia"
system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]'''

inputs = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
generated_ids = model.generate(inputs=inputs, temperature=0.7, max_new_tokens=254)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(response)