*# Evaluation Script Instructions for Google Colab*

**## Prerequisites**
Before running the evaluation script, complete the following steps:

**### 1. Modify `scrolls.py`**
Navigate to:
```
PIXIU/src/financial_evaluation/lm_eval/tasks/scrolls.py
```
Replace:
```python
from datasets import load_metric
```
with:
```python
from evaluate import load as load_metric
```

**### 2. Update Hugging Face Token**
Replace the following in the notebook:
```python
from huggingface_hub import login
login(token="your_access_token")
```
with your Hugging Face access token.

*## Running Evaluations*
To execute an evaluation for a specific model, use:
```bash
!python eval.py \
    --model "hf-causal-vllm" \
    --model_args "use_accelerate=True,pretrained='model',tokenizer='model',use_fast=False,max_gen_toks=20,dtype=float16" \
    --tasks "dataset" \
    --batch_size 30000 \
    --write_out
```

**### Selecting a Model**
To select a model from Hugging Face, navigate to "Use this model" and replace `'model'` in the command above with the appropriate model name.

Example:
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="TheFinAI/finma-7b-full")
```
Replace `'model'` in the evaluation cell with `'TheFinAI/finma-7b-full'`.


