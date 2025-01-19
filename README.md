# Anote

[Anote](https://anote.ai/) is an artificial intelligence startup in New York City, helping make AI more accessible. We are releasing **OpenAnote**, an open sourced baseline version of the Anote Product that is free to use.

### What is OpenAnote?

OpenAnote is an end to end MLOps platform that enables you to obtain the best large language model for your data. On Anote, we provide an evaluation framework to compare zero shot LLMs like GPT, Claude, Llama3 and Mistral, with fine tuned LLMs that are trained on your domain specific training data (via supervised, unsupervised and RLHF fine tuning). We provide a data annotation interface to convert raw unstructured data into an LLM ready format, and incorporate subject matter expertise into your training process to improve model accuracies. End users can route / integrate the best LLM into their own, on premise, private chatbot, as well as interact with our software development kit for fine tuning.

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/overview.png?raw=true)

OpenAnote currently supports the following task types:

- **[Text Classification](https://docs.anote.ai/classification/text-classification.html)**
- **[Named Entity Recognition](https://docs.anote.ai/ner/namedentityrecognition.html)**
- **[Chatbot](https://docs.anote.ai/privategpt/privategpt.html)** (Question Answering Across All Documents)
- **[Prompting](https://docs.anote.ai/prompting/semistructured.html)** (Question Answering Per Document)

To try the OpenAnote product for free on prod, navigate to [here](https://dashboard.anote.ai/).

### Materials on OpenAnote

| Platform         | Link                                            |
|------------------|------------------------------------------------|
| Website          | [https://anote.ai/](https://anote.ai/)         |
| Documentation    | [https://docs.anote.ai/](https://docs.anote.ai/)|
| LinkedIn         | [https://www.linkedin.com/company/anote-ai/](https://www.linkedin.com/company/anote-ai/) |
| YouTube          | [https://www.youtube.com/@anote-ai/videos](https://www.youtube.com/@anote-ai/videos) |
| Contact Email           | [nvidra@anote.ai](mailto:nvidra@anote.ai)  |
| Slack Community            | [Join Here](https://join.slack.com/t/anote-ai/shared_invite/zt-2vdh1p5xt-KWvtBZEprhrCzU6wrRPwNA)  |

### Contribution Guidelines

See the ```CODEBASE_SETUP.md``` file for codebase setup and installation, ```NAVIGATING_ANOTE_CODEBASE.md``` for understanding the how codebase work, and ```CONTRIBUTION_GUIDELINES.md``` for learning how to best contribute code. Feel free to message our team in our slack channel for any necessary technical support.

## How OpenAnote Works

OpenAnote leverages the Human Centered AI process, which consists of labeling data, training / fine tuning of large language models, making predictions across a variety of LLMs, evaluating the results of these predictions, and integrating the best model into your product:

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/HumanCentered.png?raw=true)

### 1. Create Project

To use the free version on production, navigate to [https://dashboard.anote.ai/](https://dashboard.anote.ai/) and sign in. You should get to a screen that looks like this:

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/homepage.png?raw=true)

From there, click on the top right nav to create a project. A project is a team of people that can collaborate on an AI project. Each project includes people with different roles and have access to various datasets and models. [Learn more about collaboration in projects](https://docs.anote.ai/collab/multiannotatorcollaboration.html).

### 2. Define Training and Testing Datasets

Choose ground truth testing data data that your team would like to measure model performance on, or select one of our benchmark test datasets. 

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/benchmark.png?raw=true)

*Note: Depending on the testing data chosen, we can help you curate and obtain relevant training data upon request.*

### 3. Define Evaluation Metrics

We evaluate the model accuracy and citations per task across all assessed models. To evaluate the performance of fine tuned LLMs, we have an evaluation dashboard so you see the improved model performance of supervised and unsupervised fine tuning models with metrics like Cosine Similarity, Rouge-L Score, LLM Eval, Answer Relevance and Faithfulness. Below are some evaluation metrics for each task type:

| **Task Type**                  | **Evaluation Metrics**                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Chatbot / Prompting | [Evaluation Example](https://docs.anote.ai/api-prompting/example8.html)                                   |
| Text Classification            | [Dashboard Example](https://docs.anote.ai/structured/structureddashboard.html)                                   |
| Named Entity Recognition       | [NER Evaluation](https://docs.anote.ai/structured/evaluatingner.html)                                            |

*Note: We can measure the time to call each model, and the associated cost to run each model upon request.*

### 4. Obtain Initial Predictions Results

Obtaining zero-shot results for supported LLMs.

| **Company Name** | **Model Benchmarked**        |
|------------------|------------------------------|
| OpenAI           | GPT-4o, o1                  |
| Meta             | Llama3                      |
| Anthropic        | Claude 3.5 Sonnet           |
| Google           | Gemini2                     |
| Mistral          | Mistral 7B                  |
| xAI              | Grok2                       |
| Groq             | Llama3 8B                   |

*Note: We are happy to include additional model providers upon request. We can evaluate models with enhanced RAG techniques, including Recursive Chunking, FLARE, Query Expansions, Metadata Filtering, Reranking Algorithms, and HyDE.*

### 5. Obtain Fine-Tuned Results

If the baseline zero shot results are good enough, you should be good to go. Otherwise, you can leverage our fine tuning library to do active learning and few-shot learning techniques to improve results and make higher quality predictions. We support supervised fine tuning where you can fine tune your LLM on your labeled data using parameter-efficient fine-tuning techniques like LORA and QLORA. In addition, we support RLHF / RLAIF Fine Tuning where you can add human feedback to refine LLM outputs, making them more accurate / tailored to your domain-specific needs.

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/Train.png?raw=true)

*Note: We also support Unsupervised Fine Tuning where you can upload your raw files to pre-train your LLM on domain-specific content without needing labeled / structured data, and can do this upon request.*

### 6. Exploring various labeling strategies:

Once baseline fine tuning results are established, label data to do Reinforcement Learning with Human Feedback (RLHF) to improve model performance. This process would explore various labeling strategies, including zero labels with programmatic labeling, all AI- or RLAIF-generated labels, synthetically generated labels, and datasets with N = 100, 200, 300, 400, or 500 manually labeled examples. To do the labeling, there is a 4 step process:

| **Step**      | **Description**                                                                                                                                                                           |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Upload**    | Create a new text-based dataset. This can be done by uploading unstructured data, uploading a structured dataset, connecting to data sources, scraping datasets from websites, or selecting your dataset from the Hugging Face Hub. |
| **Customize** | Choose your task type, such as Text Classification, Named Entity Recognition, or Question Answering. Add the categories, entities, or questions you care about, and insert labeling functions to include subject matter expertise.  |
| **Annotate**  | Label a few of the categories, entities, or answers to questions, and mark important features. As you annotate a few edge cases, the model actively learns and improves from human feedback to accurately predict the rest of the labels. |
| **Download**  | When finished labeling, download the resulting labels and predictions as a CSV or JSON. Export the updated model as an API endpoint to make real-time predictions on future rows of data.                                       |


![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/Label.png?raw=true)

### 7. Integrate the Best Performing Model

Each trained model will be assigned a unique model ID, which can be accessed via SDK/API for developers. You can take your exported fine tuned model, and input it into our Chatbot, your accurate enterprise AI assistant. Our Chatbot has both a UI for enterprise users and a software development kit for developers. To use the Chatbot, you can upload your documents, ask questions on your documents with LLMs like GPT, Claude, Llama2 and Mistral, and get citations for answers to mitigate the effect of hallucinations

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/Integrate.png?raw=true)


### 8. Iterate on Data and Models to Build Workflows

As you add new training data or adjust the taxonomy, you can obtain an updated model ID that is most up to date. This iterative process ensures continual improvement and alignment with your specific use cases. You can use the model ID to do custom workflows (e.g., using classification predictions with a RAG agent to answer questions).

### 9. Obtain Final Report

This report will contain the results of all the benchmarked models (fine tuned vs. zero shot) with different numbers of labels to see how each LLM performed, and measures the effect of fine tuning. In addition to traditional metrics on accuracy, time and cost, we provide metrics such as stability (how many labels are needed to be good enough) and certainty (confidence of model predictions).

![alt text](https://github.com/nv78/OpenAnote/blob/main/materials/images/Report.png?raw=true)

### 10. Obtain API Access to Model

We provide SDK / developer API access to use each of the improved model versions. Make predictions with our [fine tuning API](https://docs.anote.ai/api-anote/overview.html), which can can be used to maintain and improve models, and to integrate the best models into your product offerings. To call our pypi package on production, run
```
pip install anoteai
```
And follow the instructions within our documentation or ```CODEBASE_SETUP.md```. Once you have an API key, it should look like this:
```py
from anoteai import Anote

api_key = 'INSERT_API_KEY_HERE'
Anote = Anote(api_key)

chat_id = Anote.upload(task_type="documents", model_type="gpt4", file_paths=file_paths)['id']

response = Anote.chat(chat_id, "What does this company do?")
print(response['answer'])
print("Sources:", response['sources'])

message_id = response['message_id']
print(Anote.evaluate(message_id))
```
As an output we get:

``` py
The research paper "Improving Classification Performance With Human Feedback" is written by Eden Chung, Liang Zhang, Katherine Jijo, Thomas Clifford, and Natan Vidra.
Sources: [['Anote_research_paper.pdf', 'Improving Classification Performance With Human Feedback:\n\nLabel a few, we label the rest\n\nEden Chung, Liang Zhang, Katherine Jijo, Thomas Clifford, Natan Vidra\n\nAbstract\n\nIn the realm of artificial intelligence, where a vast majority of data is unstructured, obtaining sub-\nstantial amounts of labeled data to train supervised machine learning models poses a significant\nchallenge. To address this, we delve into few-shot and active learning, where are goal is to improve\nAI models with human feedback on a few labeled examples. This paper focuses on understanding how\na continuous feedback loop can refine models, thereby enhancing their accuracy, recall, and precision\nthrough incremental human input. '], ['Anote_research_paper.pdf', 'By employing Large Language Models (LLMs) such as GPT-3.5,\nBERT, and SetFit, we aim to analyze the efficacy of using a limited number of labeled examples to\nsubstantially improve model accuracy. We benchmark this approach on the Financial Phrasebank,Banking, Craigslist, Trec, Amazon Reviews da']]
{'answer_relevancy': 0.9307434918423216, 'faithfulness': 1.0}
```
