==============================================
Evaluation Framework Tutorial on Google Colab
==============================================

Overview
========
This tutorial explains how to use the Evaluation Framework on Google Colab to run evaluation tasks using PIXIU and DeepSeek. It also briefly covers short-task evaluations and Retrieval Augmented Generation (RAG).

Prerequisites
=============
- A Google Colab account.
- Selection of an A100 GPU runtime in Colab.
- A valid Hugging Face token with required model access.
- Required model checkpoints (e.g. BART checkpoint).

PIXIU Setup
===========
Clone the PIXIU repository and install required libraries.

Git and Library Installation
----------------------------
.. code-block:: bash

    # Upgrade torch and clone your PIXIU repository
    !pip install -U torch
    !git clone -b working <YOUR_GITHUB_REPO_LINK> --recursive
    %cd PIXIU
    !pip install -r requirements.txt

Financial Evaluation Dependencies
-----------------------------------
.. code-block:: bash

    %cd /content/PIXIU/src/financial-evaluation
    !pip install -e .[multilingual]
    !pip install bert_score
    !pip install gdown
    !pip install vllm==0.5.4
    !pip install torch==2.4.0 torchvision==0.19
    !pip install peft
    !pip install lm-eval google-generativeai

Download BART Checkpoint
------------------------
.. code-block:: python

    from google.colab import drive
    import os
    import gdown

    # Mount Google Drive
    drive.mount('/content/drive')

    source_path = "/content/drive/My Drive/bart_score.pth"
    destination_path = "/content/PIXIU/src/metrics/BARTScore/bart_score.pth"

    if os.path.exists(source_path) and not os.path.exists(destination_path):
        !cp "{source_path}" "{destination_path}"
        print("File found in Google Drive and copied.")
    else:
        file_id = '19Fpob1RhQHyvMlOqxPO89z1W58PvkOm-'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination_path, quiet=False)
        print("File not found in Google Drive. Downloaded instead.")

Set PYTHONPATH and Login
------------------------
.. code-block:: python

    %cd /content/PIXIU/src
    %cd /content

    import os
    os.environ['PYTHONPATH'] += ":/content/PIXIU/src/metrics/BARTScore/"
    !echo $PYTHONPATH

    from huggingface_hub import login
    login(token="input token")  # Replace with your Hugging Face token

Run PIXIU Evaluation Tasks
===========================
Define and run the tasks as required. Modify the task list and model parameters as needed.

.. code-block:: python

    tasks_list = [
        "flare_ner",
        "flare_finer_ord",
        "flare_finred",
        "flare_causal20_sc",
        "flare_cd",
        "flare_fnxl",
        "flare_fsrl",
        "flare_fpb",
        "flare_fiqasa",
        "flare_tsa",
        "flare_headlines",
        "flare_fomc",
        "flare_finarg_ecc_auc",
        "flare_finarg_ecc_arc",
        "flare_multifin_en",
        "flare_ma",
        "flare_mlesg",
        "flare_finqa",
        "flare_tatqa",
        "Regulations",
        "flare_convfinqa",
        "flare_edtsum",
        "flare_ectsum",
        "flare_sm_bigdata",
        "flare_sm_acl",
        "flare_sm_cikm",
        "flare_german",
        "flare_australian",
        "flare_cra_lendingclub",
        "flare_cra_ccf",
        "flare_cra_ccfraud",
        "flare_cra_polish",
        "flare_cra_taiwan",
        "flare_cra_portoseguro",
        "flare_cra_travelinsurace",
        "flare_es_financees",
        "flare_es_multifin",
        "flare_es_efp",
        "flare_es_efpa",
        "flare_es_fns",
        "flare_es_tsa"
    ]

    pretrained = "<INSERT_PRETRAINED_TRANSFORMER_URL_HERE>"
    tokenizer = "<INSERT_TOKENIZER_TRANSFORMER_URL_HERE>"
    max_gen_toks = 128
    batch_size = 20000
    num_fewshot = 0
    results_dir = "/content/results"
    model_type = "hf-causal-vllm"
    model_name = "<YOUR_MODEL_NAME>"

    os.makedirs(f"{results_dir}/{model_name}", exist_ok=True)

    for task in tasks_list:
        output_file_path = f"{results_dir}/{model_name}/{task}_results.txt"
        print(f"Running task: {task}\nSaving output to: {output_file_path}\n")

        !python PIXIU/src/eval.py \
            --model $model_type \
            --model_args "pretrained=$pretrained,tokenizer=$tokenizer,trust_remote_code=True,use_fast=False,max_gen_toks=$max_gen_toks" \
            --tasks $task \
            --batch_size $batch_size \
            --num_fewshot $num_fewshot \
            --output_base_path $results_dir \
            > $output_file_path

DeepSeek Integration
====================
DeepSeek is an additional evaluation tool that can be integrated similarly. Follow these general steps:

Clone and Setup DeepSeek
--------------------------
.. code-block:: bash

    !git clone <YOUR_DEEPSEEK_GITHUB_REPO_LINK> DeepSeek
    %cd DeepSeek
    !pip install -r requirements.txt

Configuration
-------------
- Update configuration files as needed.
- Set model and dataset parameters for DeepSeek.

Run DeepSeek Evaluation
-----------------------
.. code-block:: bash

    !python deepseek_eval.py --model <YOUR_MODEL_NAME> --data <data_path> --other_args

Short Task Evaluation and RAG
===============================
For evaluating shorter tasks or using Retrieval Augmented Generation (RAG), follow these guidelines:

Short Task Evaluation
---------------------
Utilize a dedicated script for short tasks. For example:

.. code-block:: bash

    !python short_eval.py --model <YOUR_MODEL_NAME> --data <short_task_data> --batch_size 1000

RAG Setup and Execution
-----------------------
1. Ensure that all necessary RAG libraries are installed:

   .. code-block:: bash

       !pip install transformers
       !pip install faiss-cpu

2. Run the RAG evaluation script with the proper configurations:

   .. code-block:: bash

       !python rag_eval.py --model <YOUR_MODEL_NAME> --data <data_path> --retriever <retriever_config>

Conclusion
==========
This tutorial provides a framework for running evaluation tasks on Google Colab using PIXIU and DeepSeek. Adapt the scripts and parameters as needed for your evaluation tasks, including short tasks and RAG-based experiments.
