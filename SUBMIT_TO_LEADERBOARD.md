# Submitting Your Model to the Leaderboard

Welcome to Anote! Anote is an AI startup based in New York City, dedicated to making AI accessible to everyone. The **Model Leaderboard** is a place to showcase and compare results of various AI models. When you submit your model, it will be evaluated on our benchmark datasets and added to the leaderboard, allowing you to see how it stacks up against others.

This README will guide you through downloading benchmark datasets, filling out the submission form, and completing the process to have your results added to the leaderboard.

## Table of Contents
- [What is the Model Leaderboard?](#what-is-the-model-leaderboard)
- [Prerequisites](#prerequisites)
- [Supported Task Types](#supported-task-types)
- [Submission Workflow](#submission-workflow)
- [Viewing Results](#viewing-results)

## What is the Model Leaderboard?

The **Model Leaderboard** allows researchers, developers, and organizations to showcase the performance of their AI models. We support multiple task types such as classification and question answering. By submitting your model, you contribute to transparency and progress in AI research.

- **Evaluation Process**: We will score your model using our curated benchmark datasets.
- **Timeline**: Results will be added to the leaderboard within 2-3 business days.
- **Confirmation**: Email [nvidra@anote.ai](mailto:nvidra@anote.ai) for confirmation of your submission.

## Prerequisites

### Create an Anote Account
- Visit [https://dashboard.anote.ai](https://dashboard.anote.ai) to sign up or log in.
- Once logged in, generate an API key via your account settings to enable API-based submissions.

### Model Output Requirements
- Ensure your model generates predictions for our benchmark evaluation set.
- Supported formats include JSON or CSV.

### Download Benchmark Dataset
- Navigate to the **Submit to Model Leaderboard** page.
- Select the appropriate benchmark dataset for your task type.
- Email us if you have any questions about the datasets.

## Supported Task Types

The Anote Model Leaderboard currently supports:
- **Text Classification**
- **Named Entity Recognition (NER)**
- **Chatbot (Document-Level Q&A)**
- **Prompting (Line-Level Q&A)**

Each task has curated benchmark datasets to evaluate your model's performance.

## Submission Workflow

### Submitting via the UI

1. **Navigate to Submission Page**  
   - Go to `/submittoleaderboard` on the Anote platform.
   - Download the benchmark dataset and fill out the required fields.

2. **Submit Outputs**  
   - Generate predictions for the evaluation set and upload the results.
   - Fill out the fields in the submission form
   - Ensure that the benchmark dataset name matches the provided dataset.

3. **Confirmation**  
   - Click \"Submit\" and email [nvidra@anote.ai](mailto:nvidra@anote.ai) for confirmation.
   - We will add your model to the leaderboard within 2-3 business days.
<!--
## Submit via the Anote API

If you prefer automation or command-line workflows, you can submit via our API:

1. **Install the `anoteai` Package**  
``` python
pip install anoteai
```

Obtain an API Key by signing into the Anote Dashboard. Go to your account settings or project settings to find or generate your personal API key.

### Initialize the API Client

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

``` python
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
-->
## Viewing Results
After your submission has been processed, you can view your results:

### In the UI
Check the “Model Leaderboard” tab within our website in a few business days. If requested via email, your model’s score and rank will appear alongside other submissions.
<!--
### Via the API

``` python
results = anote_client.get_leaderboard(dataset_id="DATASET_ID")
for entry in results["models"]:
    print(entry["rank"], entry["model"], entry["score"])
```
-->
Feel free to share your success on social media or in our Slack community once your model is on the leaderboard!

For guidelines on contributing code to OpenAnote, please see our ```CODEBASE_SETUP.md```, ```NAVIGATING_ANOTE_CODEBASE.md```, and ```CONTRIBUTION_GUIDELINES.md```. If you have any questions, drop them in our Slack channel or email us. Thank you for helping build a more transparent and powerful LLM community. We look forward to seeing your model on the leaderboard!
