========================================================================
Benchmarking LLama3.1 on Financial Tasks (zeroshot)
========================================================================

.. contents:: Table of Contents
   :local:

Overview
--------
This guide demonstrates a general methodology for benchmarking large language models (LLMs) on financial tasks using the `FLARE-FIQASA dataset <https://huggingface.co/datasets/ChanceFocus/flare-fiqasa>`_ and Meta's `Llama-3.2-1B model <https://huggingface.co/meta-llama/Llama-3.2-1B>`_. We will use the Zero-shot approach.

Dataset Structure
--------------------------------

The FLARE-FIQASA dataset contains financial sentiment analysis examples with the following format:

.. code-block:: json

   {
     "id": "fiqasa0",
     "query": "What is the sentiment of the following financial post: Positive, Negative, or Neutral? Text: Whats up with $LULU? Numbers looked good, not great, but good. I think conference call will instill confidence. Answer:",
     "answer": "neutral",
     "text": "Whats up with $LULU? Numbers looked good, not great, but good. I think conference call will instill confidence.",
     "choices": ["negative", "positive", "neutral"],
     "gold": 2
   }

Key fields:
- ``text``: Raw financial text
- ``choices``: Possible labels
- ``gold``: Correct answer index

Different datasets can have different structure depending on the author's design. Typically dataset authors have their designed pipelines for models evaluating their dataset, so re producing the evaluation process may give different result depending on the implementation. For demonstrations, we will be implementing an evaluation pipeline from scratch to show the evaluation process.

Implementation Steps
------------------------------------

1. Environment Setup
------------------------------------

.. code-block:: python

   pip install 'accelerate>=0.26.0' transformers datasets evaluate scikit-learn

2. Import Dependencies & Setups
------------------------------------

.. code-block:: python

    import re
    import threading
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import evaluate
    from tqdm.auto import tqdm

    # Model Variable
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "ChanceFocus/flare-fiqasa"
    ACCESS_TOKEN = "hf_token_here"
    # You must get your own Huggingface token from https://huggingface.co/settings/tokens
    # You will also need to request model access for each model you used on the huggingface model repository
    # Request access to https://huggingface.co/meta-llama/Llama-3.2-1B

3. Model Initialization
------------------------------------

.. code-block:: python

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        token=ACCESS_TOKEN,
    )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

4. Zero-Shot Inference
------------------------------------

Zero-shot: Testing without providing examples. Asking LLM your question directly.

.. code-block:: python

    # A example text we will send to the LLM
    def zero_shot_prompt(example):
        return f"""Analyze the sentiment of this financial text:
    Text: {example['text']}
    Options: {', '.join(example['choices'])}
    Answer:"""

5. Evaluation
------------------------------------

We use accuracy as our evaluation metric. There are two strategies for computing accuracy in this context:

Exact Match: The model's generated answer must exactly match the gold label. This means the extracted sentiment (e.g., positive, negative, neutral) must be identical to the expected label without any variation. If the output deviates in any way, even if it conveys the same sentiment, it is considered incorrect.

Partial Match: Instead of requiring an exact match, this approach checks whether the extracted sentiment is present within the model's response. It allows for some flexibility, ensuring that as long as the generated text contains the correct sentiment label, it is considered a correct prediction.

In our evaluation, we use partial match, as it accommodates variations in the model’s response while still capturing the intended sentiment classification.

.. code-block:: python

    def generate_with_progress(prompt, max_new_tokens):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # Create a streamer object to stream the generated text
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Using threadings to parallel generating texts.
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=max_new_tokens,
            streamer=streamer
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        with tqdm(total=max_new_tokens, desc="Generating text", unit="token", dynamic_ncols=True) as gen_bar:
            for new_text in streamer:
                generated_text += new_text
                gen_bar.update(1)
        thread.join()
        return generated_text


    def extract_answer_section(response):
        """
        Our example input looks like this
            Analyze the sentiment of this financial text:
            Text: Legal & General share price: Finance chief to step down
            Options: negative, positive, neutral
            Answer:
        Given LLM is generating word by word after the initial input. The result output will look like (For llama3.1. Can vary depending on models)
            Analyze the sentiment of this financial text:
            Text: Legal & General share price: Finance chief to step down
            Options: negative, positive, neutral
            Answer: neutral
            Explanation: xxx

        Here we extract labels after "Answer:" and before，"Explanation:"
        """
        lower_response = response.lower()
        answer_idx = lower_response.find("answer:")
        if answer_idx == -1:
            return ""
        # After "Answer:"
        answer_section = response[answer_idx + len("answer:"):].strip()
        # Before "Explanation:"
        explanation_idx = answer_section.lower().find("explanation:")
        if explanation_idx != -1:
            answer_section = answer_section[:explanation_idx].strip()
        return answer_section


    def extract_sentiment(response, choices):
        """
            Extract sentiment tags from generated text:
            1. First extract the answer part after "Answer:";
            2. Check whether the answer part contains candidate sentiment words (whole word matching, ignoring case);
            3. If there is no match, return None.
        """
        answer_section = extract_answer_section(response)
        if not answer_section:
            return None
        for choice in choices:
            pattern = r'\b' + re.escape(choice) + r'\b'
            if re.search(pattern, answer_section, re.IGNORECASE):
                return choice
        return None

    def label_to_int(label):
        """
          "negative" -> 0
          "positive" -> 1
          "neutral"  -> 2
        None -> -1。
        """
        mapping = {"negative": 0, "positive": 1, "neutral": 2}
        if isinstance(label, int):
            return label
        return mapping.get(label.lower(), -1)


    # Partial match. Print gold and generated results
    def evaluate_model(dataset_split, num_examples):
        accuracy = evaluate.load("accuracy")
        predictions = []
        references = []

        # Progress bar
        progress_bar = tqdm(
            total=num_examples,
            desc="Evaluating samples",
            unit="sample",
            dynamic_ncols=True
        )

        for i, ex in enumerate(dataset_split.select(range(num_examples))):
            inputs = tokenizer(zero_shot_prompt(ex), return_tensors="pt").to("cuda")
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True
            )
            # Decode the generated token id list into text
            response = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)

            # Use partial matching to extract sentiment labels in generated answers (only partial matching from Answer)
            extracted = extract_sentiment(response, ex['choices'])
            pred_label = extracted if extracted is not None else "unknown"

            # Print the gold answer and the full text generated for the current sample
            tqdm.write(f"Sample {i}:")
            tqdm.write(f"  Gold answer: {ex['answer']} (index: {ex['gold']})")
            tqdm.write(f"Generated text: \n\t{response}\n")
            tqdm.write(f"  Extracted LLM answer: {pred_label}")
            tqdm.write("-" * 60)

            # Convert predictions and gold labels to integers
            predictions.append(label_to_int(pred_label))
            references.append(label_to_int(ex['gold']))

            progress_bar.update(1)
            current_acc = accuracy.compute(predictions=predictions, references=references)['accuracy']
            progress_bar.set_postfix({"current_acc": f"{current_acc:.2%}"})

        progress_bar.close()
        return accuracy.compute(predictions=predictions, references=references)

6. Usage
------------------------------------

.. code-block:: python

    dataset = load_dataset(DATASET_NAME)

    example = dataset["test"][0]
    zero_prompt = zero_shot_prompt(example)
    print("\nRunning zero-shot inference:")
    response = generate_with_progress(zero_prompt, max_new_tokens = 10)
    extracted_label = extract_sentiment(response, example['choices'])
    print(f"\nZero-Shot Extracted Label: {extracted_label if extracted_label is not None else 'unknown'}")

    print("\nStarting evaluation:")
    results = evaluate_model(dataset["test"], num_examples=10)
    print(f"\nFinal Accuracy: {results['accuracy']:.2%}")

