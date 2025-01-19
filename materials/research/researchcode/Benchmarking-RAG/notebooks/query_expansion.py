import pandas as pd
from openai import OpenAI
import chromadb
from datasets import load_dataset
import requests
import os
import glob
from pypdf import PdfReader

import os
root = os.getcwd().split('Anote')[0] + 'Anote'
path_to_pdf_storage = f'{root}/Benchmarking_RAG/documents'


def extract_pdf_url(url):
    """
    Extracts the actual PDF URL from the given URL.
    Decodes it from base64 if necessary.
    """
    if url.lower().endswith('.pdf'):
        return url  # Direct PDF URL
    else:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        pdf_target = query_params.get('pdfTarget', [None])[0]

        if pdf_target:
            pdf_url = base64.b64decode(pdf_target).decode('utf-8')
            return pdf_url
        else:
            raise ValueError("No valid PDF URL found in the provided URL")


def download_pdf(url, save_path):
    """
    Downloads a PDF from a given URL.
    """
    try:
        pdf_url = extract_pdf_url(url)
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Ensure the request was successful
        if not(os.path.exists(save_path)):
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Downloaded PDF from: {pdf_url} to {save_path}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")

def get_pages_from_pdf(path):
    reader = PdfReader(path)
    pages_text = []
    for idx, page in enumerate(reader.pages):
        pages_text.append(page.extract_text())
    return pages_text

def seed_chroma():
    client = chromadb.PersistentClient(path=f'{path_to_pdf_storage}/chromadb.db')
    collection_name = "FinanceBench_Embeddings"
    if not(collection_name in [c.name for c in client.list_collections()]):
        collection = client.create_collection(name=collection_name)#, embedding_function=embedding_function)
    else:
        print('already exists - returning')
        return

    files = glob.glob(f'{path_to_pdf_storage}/*.pdf')
    files = [x for x in files if not(".pdf.pdf" in x)]
    print('files is:', files)
    for idx, path in enumerate(files):
        pages = get_pages_from_pdf(path)
        my_ids=[f'{str(idx)}_{x[0]}' for x in list(enumerate(pages))]
        collection.add(
            documents= pages,
            ids=my_ids,
            metadatas=[{'doc_path':path}]*len(my_ids),

        )




def no_context(query, openai_api_key):
    ai_model = "gpt-4o"
    client = OpenAI(api_key=openai_api_key)

    system_prompt ="""You are a financial chatbot trained to answer financial questions to the absolute best of your ability. Your primary focus should be on accuracy, specificity, and correctness, particularly relating to financial statements, company performance, and market position. Please answer each question with total accuracy, performing all necessary calcualtions without skipping or simplying any steps along the way. If you do not have enough information to answer a question, please make whatever reasonable assumptions are necessary and provide a full and complete answer."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role":"user", "content":f"Question: {query}"},
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = messages
    )
    return completion.choices[0].message.content

def with_initial_response(query, initial_response, openai_api_key):
    collection_name="FinanceBench_Embeddings"
    chroma_client = chromadb.PersistentClient(path=f'{path_to_pdf_storage}/chromadb.db')
    collection = chroma_client.get_collection(collection_name)
    rag_text = '\n\n'.join(collection.query(query_texts=[query, initial_response], n_results=10)['documents'][0])

    system_prompt = """You are a financial chatbot trained to answer questions based on the information provided in 10-Ks and other financial
    documents. Your responses should be directly sourced from the content of these documents. When asked
    a question, ensure that your answer is explicitly supported by the text in the document, and do not
    include any external information, interpretations, or assumptions not clearly stated in the document. If
    a question pertains to financial data or analysis that is not explicitly covered in the document text filing provided,
    respond by stating that the information is not available in the document. Your primary focus should
    be on accuracy, specificity, and adherence to the information in the documents, particularly regarding
    financial statements, company performance, and market position."""

    query_prompt = f"Question: {query}. Relevant document information: {rag_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role":"user", "content":query_prompt},
    ]
    openai_client = OpenAI(api_key=openai_api_key)
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages = messages
    )
    return completion.choices[0].message.content


def main(openai_api_key):
    dataset = load_dataset("PatronusAI/financebench")
    df = pd.DataFrame(dataset['train'])
    download_dir = "documents"
    if not(os.path.exists(download_dir)):
           os.makedirs(download_dir, exist_ok=True)
           df.apply(lambda x: download_pdf(x.doc_link, os.path.join(download_dir, f"{x.doc_name}.pdf")), axis=1) #download all the pdfs
           seed_chroma()
    for index, row in df.iterrows():
        query = row['question']
        # refrence_answer = row['answer']
        # doc_name = row['doc_name']
        # doc_link = row['doc_link']

        initial_response = no_context(query, openai_api_key)
        final_response = with_initial_response(query, initial_response, openai_api_key)

        print(final_response)



def evaluate_model(test, openai_api_key):
    results_dl = []
    for index, row in test.iterrows():
        download_dir = "pdf_documents"
        os.makedirs(download_dir, exist_ok=True)
        doc_link = row['doc_link']
        doc_name = row['doc_name']
        query = row['question']
        ref_answer = row['answer']
        ref_context = row['evidence_text']
        doc_path = os.path.join(download_dir, f"{doc_name}.pdf")

        download_pdf(doc_link, doc_path)
        create_chroma_vectordb_from_pdf(doc_path, openai_api_key)
        print("Querying Model now")
        initial_response = no_context(query, openai_api_key)
        final_response = with_initial_response(query, initial_response, openai_api_key)


        #Evaluation
        cosine_similarity = cosine_similarity(final_response, ref_answer)
        bert_score = bert_score(final_response, ref_answer)
        llm_eval = evaluate_llm_responses(question, final_response, ref_answer)


        results_dl.append({
            'doc_name': doc_name,
            'question': question,
            'ref_answer': ref_answer,
            'final_response': final_response,
            'cosine_similarity': cosine_similarity,
            'bert_score': bert_score,
            'llm_eval': llm_eval
        })
    results_df = pd.DataFrame(results_dl)
    results_df.to_csv('query_expansion_results.csv', index=False)
    return results_df