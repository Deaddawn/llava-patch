from transformers import AutoTokenizer
from llava.model import *

tokenizer=AutoTokenizer.from_pretrained('/root/szd2/LLaVA/ckpt/vicuna-7b-1.5')
model=LlavaLlamaForCausalLM.from_pretrained('/root/szd2/LLaVA/ckpt/vicuna-7b-1.5')


tokenizer.add_tokens(['<box>','<abox>'],special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained('/root/szd2/LLaVA/ckpt/vicuna-7b-1.5-altered')
model.save_pretrained('/root/szd2/LLaVA/ckpt/vicuna-7b-1.5-altered')