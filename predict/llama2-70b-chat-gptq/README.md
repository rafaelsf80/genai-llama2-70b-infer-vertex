Here the model to be downloaded from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-70B-chat-GPTQ/tree/main). 
No `handler.py` required since we will not use TorchServe. Note also the size (2 GiB since it is a 2-bit GGML model):
```sh
config.json
generation_config.json
model.safetensors
quantize_config.json
special_tokens_map.json
tokenizer.json
tokenizer.model
tokenizer_config.json
```