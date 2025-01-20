# Submitting Your Model to the OpenAnote Leaderboard

Welcome to **Anote**—we are an AI startup based in New York City, dedicated to making AI accessible for everyone. Our free and open-sourced platform, **OpenAnote**, is an MLOps solution that helps you:

- Label and prepare data for Large Language Models (LLMs).  
- Fine-tune LLMs on domain-specific data (using supervised, unsupervised, or RLHF approaches).
- Evaluate zero-shot performance of LLMs like GPT, Claude, Llama3, Mistral, and more.  

The **Model Leaderboard** is where we showcase and compare results for various LLMs. If you’ve improved an LLM or built a new one, this README will guide you through submitting it to our leaderboard so the community can see how your model stacks up!

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Task Types We Support](#task-types-we-support)  
3. [Submit via the OpenAnote UI](#submit-via-the-openanote-ui)  
4. [Submit via the Anote API](#submit-via-the-anote-api)  
5. [Viewing Results](#viewing-results)  

---

## 1. Prerequisites

- **Create an Anote Account**  
Head to [https://dashboard.anote.ai/](https://dashboard.anote.ai/) to sign up or log in. You will gain access to your personal workspace and can create an API key.

We only require the outputs your model generates for each query in our evaluation set. You can use our API / SDK, or can leverage a custom LLM for fine tuning.

If you need help obtaining our benchmark data or have questions about which benchmark suits your task, reach out to us at [nvidra@anote.ai](mailto:nvidra@anote.ai).

---

## 2. Task Types We Support

OpenAnote currently supports:

1. **Text Classification**  
2. **Named Entity Recognition (NER)**  
3. **Chatbot (Question Answering Across All Documents)**  
4. **Prompting (Question Answering Per Document)**  

For each of these task types, we maintain curated benchmark datasets that we measure on our model leaderboard.

---

## 3. Submit via the OpenAnote UI

1. **Navigate to Submission Page**  
   Click the [Submit to Model Leaderboard](https://anote.ai/leaderboard) button here, and sign into Anote. This will navigate you to the `/submittoleaderboard` page.

2. **Select a Benchmark Dataset**  
   - If you’re using an existing benchmark dataset, simply select it from the “Datasets” section.
   - Input the relevant fields on the form.  

3. **Upload Your Model Outputs**  
   - Download our evaluation set from the UI, or email us for direct access if needed.  
   - Generate predictions or responses for each query in the set.  
   - Upload these predictions (as a CSV file).  

5. **Confirm and Submit**  
   - Verify that your submission is tied to the correct dataset.  
   - Click “Submit” or “Upload Outputs” to send your predictions to our scoring pipeline.

6. **Wait for Evaluation**  
   Our system will process and score your model in a few business days. We’ll notify you once it’s complete.

---

## 4. Submit via the Anote API

If you prefer automation or command-line workflows, you can submit via our API:

1. **Install the `anoteai` Package**  
   ``` python
   pip install anoteai
   ```
Obtain an API Key

Sign in to Anote Dashboard.
Go to your account settings or project settings to find or generate your personal API key.
Initialize the API Client

```python
from anoteai import Anote

api_key = 'INSERT_API_KEY_HERE'
anote_client = Anote(api_key)
```

### Download the Evaluation Set
You can either do this through the UI or via the API. The exact endpoint will be documented in our docs.

### Generate Predictions
Using your own model, produce outputs for each query in the evaluation set. Format them as required (often JSON lines, one per sample).

### Submit Predictions

```python
# Example: posting model outputs (pseudo-code)
submission_response = anote_client.submit_predictions(
    dataset_id="DATASET_ID",
    task_type="text_classification",   # or "ner", "chatbot", "prompting"
    model_name="MyCustomModel-v1",
    predictions_file="/path/to/predictions.json"
)
print(submission_response)
# dataset_id will correspond to the specific benchmark dataset you’re targeting.
# model_name is how you want your model labeled on the leaderboard.
# Check Status

# Retrieve submission status
status_response = anote_client.get_submission_status(submission_response["submission_id"])
print(status_response)
# You will see whether your submission is “Processing,” “Complete,” or if there were errors.
```

## 5. Viewing Results
After your submission has been processed, you can view your results:

### In the UI
Check the “Model Leaderboard” or “Evaluation Results” tab within our website in a few business days. If requested via email, your model’s score and rank will appear alongside other submissions.

### Via the API

`` python
results = anote_client.get_leaderboard(dataset_id="DATASET_ID")
for entry in results["models"]:
    print(entry["rank"], entry["model"], entry["score"])
```
Feel free to share your success on social media or in our Slack community once your model is on the leaderboard!

For guidelines on contributing code to OpenAnote, please see our CODEBASE_SETUP.md, NAVIGATING_ANOTE_CODEBASE.md, and CONTRIBUTION_GUIDELINES.md. If you have any questions, drop them in our Slack channel or email us.

Thank you for helping build a more transparent and powerful LLM community. We look forward to seeing your model on the leaderboard!
