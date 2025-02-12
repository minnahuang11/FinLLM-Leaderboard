.. _zero_shot_openai:

Zero-Shot Benchmark Tutorial (OpenAI Only)
==========================================

This guide shows how to benchmark an OpenAI model in a **Zero-Shot** setting:

1. **Install** the necessary libraries.  
2. **Load** your OpenAI API key.  
3. **Prompt** the model in a zero-shot manner for each question in a CSV file.  
4. **Save** model outputs in a new column named ``generated_answer`` in the resulting CSV.

Prerequisites
-------------

1. **config.json** containing your keys. For this tutorial, only ``openai_api_key`` is actually used, but you can also store other keys for future expansions:

   .. code-block:: json

       {
         "openai_api_key": "sk-your-openai-key-here",
         "mistral_api_key": "ignored_for_this_tutorial",
         "together_ai_api_key": "ignored_for_this_tutorial"
       }

   Only the ``openai_api_key`` is used in **this** example, but the others are loaded in case you want to integrate Mistral or LLaMA later.

2. A CSV file named ``questions.csv`` with at least a ``question`` column, columns for answer choices, and a column for the expected answer. Here's an example.:

   .. list-table:: questions.csv
      :header-rows: 1
      :widths: 20 20 20 20 20

      * - question
        - choiceA
        - choiceB
        - choiceC
        - expected_answer
      * - "Sammy Sneadle, CFA, is the founder of the Everglades Fund. The question is: how did he violate the standard by not disclosing back-tested data?"
        - "A. He did not disclose the use of back-tested data."
        - "B. He failed to deduct fees before returns."
        - "C. He did not show a weighted composite of similar portfolios."
        - "A"

3. Install the following libraries:

   .. code-block:: bash

      pip install openai==0.28.0 \
                  pandas \
                  tqdm \
                  torch \

Tutorial
--------

1. Import Libraries

   .. code-block:: python

      import os
      import json
      import pandas as pd
      from tqdm import tqdm
      import openai
      import torch

2. Load Keys

   .. code-block:: python
      
      with open("config.json", "r", encoding="utf-8") as f:
          config = json.load(f)

      # Required for OpenAI
      openai.api_key = config["openai_api_key"]

2. Define a Zero-Shot Inference Function

   .. code-block:: python

      def generate_multiple_choice_response(
          prompt,
          choiceA,
          choiceB,
          choice C,
          system_prompt =  (
              "You are a CFA (chartered financial analyst) taking a test to "
              "evaluate your knowledge of finance. You will be given a question along "
              "with three possible answers (A, B, and C). Provide only the letter "
              "for the correct choice (A, B, or C)."
              ),
          model_name = "gpt-4",
          temperature = 0,
          max_tokens = 128,
        ):
          """
          Generate a zero-shot response using OpenAI's ChatCompletion API.

          :param str prompt: The actual question from the user. In zero-shot, the user doesn't give any examples.
          :param str choiceA: Choice A for the question
          :param str choiceB: Choice B for the question
          :param str choiceC: Choice C for the question
          :param str system_prompt: General instructions or domain expertise the model follows
          :param str model_name: LLM name, e.g. "gpt-3.5-turbo" or "gpt-4"
          :param float temperature: Affects randomness of output (0.0 = effectively deterministic)
          :param int max_tokens: Adds a cap to the length of the response the LLM can generate

          Returns:
              str: The model's generated answer
          """
          try:
              # Construct a user prompt with the question and answer choices
              # A user prompt is the actual question while the system prompt tells the model how to behave
              user_prompt = (
                f"Question:\n{question}\n\n"
                f"{choiceA}\n{choiceB}\n{choiceC}\n\n"
                "Which choice is correct (A, B, or C)?"
              )
              response = openai.ChatCompletion.create(
                  model=model_name,
                  messages=[
                      {"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}
                  ],
                  temperature=temperature,
                  max_tokens=max_tokens
              )
              # Get the generated answer
              return response["choices"][0]["message"]["content"].strip()
          except Exception as e:
              print(f"Error: {e}")
              return "Error generating response"

3. Read question from CSV file, prompt the model, and save results

   .. code-block:: python

      def run_zero_shot(
          input_csv = "questions.csv",
          output_csv = "gpt_4_answers.csv",
          model_name = "gpt-4"
      ):
          """
          Reads a CSV of questions, prompts GPT-4 to answer the question in the zero-shot setting,
          and writes answers to 'generated_answer' in the output CSV.
            :param str input_csv: Path to the CSV file containing questions
            :param str output_csv: Path where the CSV file with the model's answers is saved.
            :param str model_name: The model we are testing (default is "gpt-4:)
          """
          # Load questions
          df = pd.read_csv(input_csv)

          # For each question, generate an answer
          answers = []

          # This allows us to show the progress of the model in answering questions
          for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating MC Answers"):
              q_text = row["question"]
              a_text = row["choiceA"]
              b_text = row["choiceB"]
              c_text = row["choiceC"]

              # Generate response in zero-shot. Again, zero-shot means the model is provided the question without any examples.
              # Generate the model's letter
              answer = generate_multiple_choice_response(
                  question=q_text,
                  choiceA=a_text,
                  choiceB=b_text,
                  choiceC=c_text,
                  model_name=model_name,
                  temperature=0.0,
                  max_tokens=128
              )

              # In case the model says "A. answer choice text" or "A is correct."
              answer = answer.strip().upper()[:1]

              # Add the answer to the answer array.
              answers.append(answer)

          # Add a column for generated answers to the dataframe. So the dataframe contains questions and generated answers.
          df["generated_answer"] = answers
          df.to_csv(output_csv, index=False)
          print(f"Results saved to {output_csv}")

5. Score the model's responses

   .. code-block:: python

      def score_multiple_choice(
          input_csv = "gpt_4_answers.csv",
          output_csv= "gpt_4_answers_scored.csv"
      ):
          """
          Loads the CSV with 'generated_answer' and 'expected_answer'.
          Scores the answer as 'T' if correct or 'F' if incorrect.
          :param str input_csv: Path to the CSV file containing the model's answers
          :param str output_csv: Path where the CSV file with the model's answers scored.
          """
          df = pd.read_csv(input_csv)

          if "generated_answer" not in df.columns:
              print("No 'generated_answer' column found. Cannot score.")
              return

          if "expected_answer" not in df.columns:
              print("No 'expected_answer' column found. Cannot score.")
              return

          accuracy_scores = []
          for _, row in df.iterrows():
              expected = str(row["expected_answer"]).strip().upper()
              generated = str(row["generated_answer"]).strip().upper()

              if generated == expected:
                  accuracy_scores.append('T')
              else:
                  accuracy_scores.append('F')

          df["accuracy_score"] = accuracy_scores
          df.to_csv(output_csv, index=False)
          print(f"Scored results saved to {output_csv}")

4. Run the inference and scoring

   .. code-block:: python

      if __name__ == "__main__":
            # 1. Run the multiple-choice question inference
            run_multiple_choice(
                input_csv="questions.csv",
                output_csv="gpt_4_answers.csv",
                model_name="gpt-4"
            )

            # 2. Score the model responses
            score_multiple_choice(
                input_csv="gpt_4_answers.csv",
                output_csv="gpt_4_answers_scored.csv"
            )

Running the Tutorial
--------------------

1. Make sure ``config.json`` contains your OpenAI API key.
2. Place your questions in ``questions.csv``. 
3. Install dependencies as above.
4. Add all the code to one file called ``zero_shot.py``.
5. Run ``python zero_shot.py``

Notes taht you can refer back to later
--------------------------------------

- **Zero-shot** means that you prompt a model with just the question and no examples.
- **temperature** determines the randomness of the model response. The closer to 0, the more deterministic and consistent the model response is.
- **max_tokens** determines the maximum length of the output of the model.
- **system_prompt** determines the behavior/domain context the model should follow.
- **prompt** is the actual question the user gives to the model. It is also called a user prompt.