=============
Introduction
=============

.. contents:: Table of Contents
   :local:

Description
============
Welcome to the `Open FinLLM Leaderboard <https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard>`_!

The OpenFinLLM Leaderboard provides an evaluation framework tailored for financial language models. Through comprehensive benchmarking of 30 LLMs across about 50 financial tasks, we aim to help researchers and practitioners identify the right model for their financial applications.

Our platform offers:

- **Comprehensive Evaluation**: Detailed assessment across seven key financial categories
- **Real-World Relevance**: Benchmarks based on actual financial industry challenges
- **Zero-Shot Testing**: Evaluation of models' ability to generalize to unseen financial tasks
- **Transparent Metrics**: Clear performance metrics for informed model selection

.. image:: ./images/overview.png
   :align: center
   :class: custom-img

Motivation
===========
The growing complexity of financial language models necessitates evaluations that go beyond general NLP benchmarks. While traditional leaderboards focus on broader NLP tasks, they often fall short in addressing the specific needs of the finance industry.

Our goal is to fill this critical gap by providing:

- A transparent framework for assessing model readiness in real-world financial applications
- Specialized evaluation metrics that matter most to finance professionals
- Clear insights into model performance across different financial tasks
- A platform for continuous improvement and innovation in financial AI

Key Features
============
Task Categories
------------------
The leaderboard evaluates models across seven essential categories:

- Information Extraction (IE)
- Textual Analysis (TA)
- Question Answering (QA)
- Text Generation (TG)
- Risk Management (RM)
- Forecasting (FO)
- Decision-Making (DM)

Each category is designed to assess specific capabilities required in financial applications, from extracting information from regulatory filings to predicting market trends.

Evaluation Metrics
------------------
We employ diverse metrics to provide a comprehensive assessment:

- F1-Score: For balanced evaluation of classification tasks
- Accuracy: For overall performance measurement
- RMSE: For quantitative prediction tasks
- Entity F1 Score: For entity recognition tasks
- ROUGE Score: For text generation evaluation
- Matthews Correlation Coefficient: For binary classification tasks
- Sharpe Ratio: For risk-adjusted return measurement