import base64
import json
import re
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from urllib.parse import parse_qs, urlparse
import requests
import chromadb
import openai
from chromadb.config import Settings
import fitz  # PyMuPDF
from chromadb.utils import embedding_functions
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score

def extract_pdf_url(url):
    """
    Extracts the actual PDF URL from the given URL.
    Decodes it from base64 if necessary.
    """
    if url.lower().endswith('.pdf'):
        return url  # Direct PDF URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    pdf_target = query_params.get('pdfTarget', [None])[0]

    if pdf_target:
        pdf_url = base64.b64decode(pdf_target).decode('utf-8')
        return pdf_url
    raise ValueError("No valid PDF URL found in the provided URL")

def download_pdf(url, save_path):
    """
    Downloads a PDF from a given URL.
    """
    try:
        pdf_url = extract_pdf_url(url)
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Ensure the request was successful

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded PDF from: {pdf_url} to {save_path}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def create_chroma_vectordb_from_pdf(pdf_path, openai_api_key, batch_size=100):
    """
    Creates a Chroma vector database from a PDF.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        sentences = text.split('\n')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )

        vectors = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            if batch:  # Ensure batch is not empty
                batch_vectors = openai_ef(batch)
                vectors.extend(batch_vectors)

        client = chromadb.Client(Settings())

        collection_name = "Finance_bench_documents"
        collection = client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

        for i, (sentence, vector) in enumerate(zip(sentences, vectors)):
            collection.upsert(f"id_{i}", vector, {"sentence": sentence})

        print(f"Stored {len(sentences)} vectors in the Chroma vector database.")
        return collection # Return the created collection
    except Exception as e:
        print(f"Error creating Chroma vector database: {e}")

def display_chat_history(messages):
    for message in messages:
        print(f"{message['role'].capitalize()}: {message['content']}")

def get_assistant_response(messages, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    #print(response)
    return response.choices[0].message.content

def query_openai_with_context(query, openai_api_key, collection, top_k=5):
    """
    Queries OpenAI with context retrieved from Chroma.
    """
    try:
        collection = collection

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key)
        query_vector = openai_ef(query)
        results = collection.query(query_texts = [query], n_results=top_k)
        #print(results)
        context = ""
        for metadata_pair in results['metadatas']:
            for metadata in metadata_pair:
                context += f"{metadata['sentence']}\n"

        print("Context retreived:", context)
        template = """You are a financial chatbot trained to answer questions based on the information provided in 10-K
        documents. Your responses should be directly sourced from the content of these documents."""

        prompt = f"\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"

        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
        response = get_assistant_response(messages, openai_api_key)

        return response
    except Exception as e:
        print(f"Error querying OpenAI with context: {e}")
        return ""

def calculate_cosine_similarity(text1, text2):
    """
    Calculates cosine similarity between two texts.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = cosine_sim[0][0]
    return similarity_score

def calculate_bertscore(candidate, reference):
    """
    Calculates BERTScore for the candidate and reference texts.
    """
    P, R, F1 = score([candidate], [reference], lang="en", verbose=True)
    return P.mean().item()

def evaluate_llm_responses(question, model_answer, reference_answer, openai_api_key):
    """
    Evaluates LLM responses against reference answers.
    """
    evaluation_scores = []

    evaluation_prompt = f"""
    Evaluate the following response against the reference answer. Assign a score between 0 and 1 based on correctness and provide a brief justification.

    Question: {question}
    Response: {model_answer}
    Reference Answer: {reference_answer}

    Score (0 to 1):
    Justification:
    """
    messages = [
        {"role": "system", "content": "You are an evaluator that scores responses based on correctness."},
        {"role": "user", "content": evaluation_prompt}
    ]
    evaluation_response = get_assistant_response(messages, openai_api_key)

    evaluation_text = evaluation_response.strip()
    print(f"Evaluation Response: {evaluation_text}")  # Print the entire response for debugging

    try:
        # Use regular expression to extract the score
        match = re.search(r'Score\s*\(\s*0\s*to\s*1\s*\)\s*:\s*([0-9]*\.?[0-9]+)', evaluation_text)
        if match:
            score = float(match.group(1))
            evaluation_scores.append(score)
        else:
            raise ValueError("Score not found in the response")
    except Exception as e:
        print(f"Error parsing score: {e}")
        evaluation_scores.append(0.0)

    average_score = sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0
    print(f'Average Correctness Score: {average_score:.2f}')
    return average_score

def main():
    """
    Main function to run the PDF-based RAG evaluation.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    dataset = load_dataset("PatronusAI/financebench")
    df = pd.DataFrame(dataset['train'])
    test = df[:5]
    results_list = []

    for index, row in test.iterrows():
        download_dir = "pdf_documents"
        os.makedirs(download_dir, exist_ok=True)
        doc_link = row['doc_link']
        doc_name = row['doc_name']
        question = row['question']
        ref_answer = row['answer']
        ref_context = row['evidence_text']
        doc_path = os.path.join(download_dir, f"{doc_name}.pdf")

        download_pdf(doc_link, doc_path)
        collection = create_chroma_vectordb_from_pdf(doc_path, openai_api_key)
        print("Querying Model now")
        model_answer = query_openai_with_context(question, openai_api_key,collection)
        print(model_answer)

        cosine_similarity_score = calculate_cosine_similarity(model_answer, ref_answer)
        bert_score = calculate_bertscore(model_answer, ref_answer)
        llm_eval = evaluate_llm_responses(question, model_answer, ref_answer, openai_api_key)

        results_list.append({
            'doc_name': doc_name,
            'question': question,
            'ref_answer': ref_answer,
            'model_answer': model_answer,
            'cosine_similarity': cosine_similarity_score,
            'bert_score': bert_score,
            'llm_eval': llm_eval
        })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('results_test.csv', index=False)

if __name__ == "__main__":
    main()
