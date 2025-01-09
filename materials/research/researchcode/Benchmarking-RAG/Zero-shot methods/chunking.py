import os
import requests
import fitz  # PyMuPDF
from urllib.parse import parse_qs, urlparse
import base64
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import pandas as pd
from dotenv import load_dotenv
from rich import print
from openai import OpenAI, ChatCompletion
import uuid
from langchain_openai.embeddings import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

# Ensure this import is after chromadb is fully imported
try:
    from chromadb.utils import embedding_functions
except AttributeError as e:
    print(f"Error importing embedding_functions: {e}")

# Load environment variables
load_dotenv()

def extract_pdf_url(url):
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

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

from transformers import GPT2Tokenizer
def truncate_chunks(sentences, max_tokens = 8192):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence))
        if current_length + sentence_length > (max_tokens):
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def process_and_combine(lst, sublist_size, sublist_func):
    sublists = [lst[i:i + sublist_size] for i in range(0, len(lst), sublist_size)]
    processed_sublists = [sublist_func(sublist) for sublist in sublists]
    combined_result = [item for sublist in processed_sublists for item in sublist]
    return combined_result

#cleaning up inputs into the embedding function
import re  
def ensure_utf8(strings):
    cleaned_strings = []
    for string in strings:
        clean_string = string.encode('utf-8', 'replace').decode('utf-8', 'replace')
        clean_string = re.sub(r'<\|.*?\|>', '', clean_string)
        clean_string = ' '.join(clean_string.split())
        cleaned_strings.append(clean_string)
    return cleaned_strings
import nltk
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
openai_api_key = ""
def create_embeddings(documents, openai_api_key, max_tokens = 1000):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002"
    )
    sentences = [doc.page_content for doc in documents]
    new_sentences = []
    page_connection = []
    for i, sentence in enumerate(sentences):
        if sentence == "" or "\n":
            continue
        if len(tokenizer.encode(sentence)) > (max_tokens):
            new_sentences = nltk.sent_tokenize(sentence)
            new_chunks = truncate_chunks(new_sentences, max_tokens=max_tokens)
            for chunk in new_chunks:
                if len(tokenizer.encode(chunk)) > 1000:
                    first_half=chunk[:(len(chunk)//2)]
                    second_half=chunk[:(len(chunk)//2)]
                    new_sentences.append(first_half)
                    page_connection.append(first_half)
                    new_sentences.append(second_half)
                    page_connection.append(second_half)
                else:
                    new_sentences.append(chunk)
                    page_connection.append(chunk)
        else:
            chunk = sentence
            first_half=chunk[:(len(chunk)//2)]
            second_half=chunk[:(len(chunk)//2)]
            new_sentences.append(first_half)
            page_connection.append(first_half)
            new_sentences.append(second_half)
            page_connection.append(second_half)
    new_sentences = ensure_utf8(new_sentences)
    new_sentences = [x.replace("\n", " ").replace(' ', '') for x in new_sentences]
    vectors = process_and_combine(new_sentences, 2047, openai_ef)
    vectors_pages = list(zip(vectors, page_connection))
    return vectors_pages

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

class AgenticChunker:
    def __init__(self, openai_api_key):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True
        openai.api_key = openai_api_key

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return
        chunk_id = self._find_relevant_chunk(proposition)
        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("No chunks found")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
        A new proposition was just added to one of your chunks, generate a brief 1-sentence summary for the chunk.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Chunk's propositions:\n{'\n'.join(chunk['propositions'])}\n\nCurrent chunk summary:\n{chunk['summary']}"}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0.0
        )
        new_chunk_summary = response.choices[0].message['content'].strip()
        return new_chunk_summary

    def _update_chunk_title(self, chunk):
        prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
        A new proposition was just added to one of your chunks, generate a brief updated chunk title.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Chunk's propositions:\n{'\n'.join(chunk['propositions'])}\n\nChunk summary:\n{chunk['summary']}\n\nCurrent chunk title:\n{chunk['title']}"}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0.0
        )
        updated_chunk_title = response.choices[0].message['content'].strip()
        return updated_chunk_title

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)
        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def _get_new_chunk_summary(self, proposition):
        prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
        Generate a brief 1-sentence summary for the new chunk.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Proposition:\n{proposition}"}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0.0
        )
        new_chunk_summary = response.choices[0].message['content'].strip()
        return new_chunk_summary

    def _get_new_chunk_title(self, new_chunk_summary):
        prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
        Generate a brief title for the new chunk.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Chunk summary:\n{new_chunk_summary}"}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0.0
        )
        new_chunk_title = response.choices[0].message['content'].strip()
        return new_chunk_title

    def _find_relevant_chunk(self, proposition):
        # This is a placeholder for the actual logic to find a relevant chunk
        return None

    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = [" ".join(chunk['propositions']) for chunk in self.chunks.values()]
            return chunks

    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

def basic_chunk(text):
    sentences = nltk.sent_tokenize(text)
    paragraphs = []
    current_paragraph = []
    for sentence in sentences:
        if sentence.strip() == "":  # Check for empty lines indicating paragraph breaks
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(sentence)
    if current_paragraph:  # Add any remaining sentences as the last paragraph
        paragraphs.append(" ".join(current_paragraph))
    documents = [Document(page_content=paragraph) for paragraph in paragraphs]
    return documents

def chunk_text(text, method="character", chunk_size=100, chunk_overlap=0):
    if method == "character":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif method == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif method == "semantic":
        text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=""), breakpoint_threshold_type="percentile")
        documents = text_splitter.create_documents([text])
        return documents
    elif method == "agentic":
        ac = AgenticChunker(openai_api_key="")
        sentences = text.split('.')
        ac.add_propositions(sentences)
        chunks = ac.get_chunks(get_type='list_of_strings')
        documents = [Document(page_content=chunk) for chunk in chunks]
    elif method == "basic":
        documents = basic_chunk(text)
        return documents
    else:
        raise ValueError("Unknown chunking method")
    return text_splitter.create_documents([text])

def store_embeddings_in_chroma(documents, vectors, collection_name="Finance_bench_documents"):
    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(name=collection_name)
    for i, (vector, page_connection) in enumerate(vectors):
         collection.upsert(f"id_{i}", vector, {"sentence": page_connection})
local_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key = "")

def rag(question, collection_name="FinanceBench_Embeddings", api_key=""):
    client = chromadb.Client(Settings())
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )
    collection = client.get_collection(collection_name, embedding_function=openai_ef)
    top_vectors = collection.query(query_texts=[question], n_results=10)
    rag_text = ""
    for metadata_pair in top_vectors['metadatas']:
        for metadata in metadata_pair:
            rag_text += f"{metadata['sentence']}\n"
    if len(rag_text) > 128000:
        rag_text = rag_text[:128000]

    template = """You are a financial chatbot trained to answer questions based on the information provided in 10-K
    documents. Your responses should be directly sourced from the content of these documents. When asked
    a question, ensure that your answer is explicitly supported by the text in the 10-K filing, and do not
    include any external information, interpretations, or assumptions not clearly stated in the document. If
    a question pertains to financial data or analysis that is not explicitly covered in the 10-K filing provided,
    respond by stating that the information is not available in the document. Your primary focus should
    be on accuracy, specificity, and adherence to the information in 10-K documents, particularly regarding
    financial statements, company performance, and market position."""

    prompt_template = ChatPromptTemplate.from_template(template)

    query_prompt = f"Question: {template}. Relevant document information: {rag_text}"
    messages = [
        # {"role": "system", "content": system_prompt},
        {"role":"user", "content":query_prompt},
    ]
    openai_client = OpenAI(api_key=openai_api_key)
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages = messages
    )
    return completion.choices[0].message.content

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = cosine_sim[0][0]
    return similarity_score

def calculate_bertscore(candidate, reference):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=True)
    return P.mean().item()

from langchain.schema import HumanMessage, SystemMessage
def evaluate_llm_responses(question, model_answer, reference_answer):
    evaluation_scores = []
    messages = [
        SystemMessage(content="You are an evaluator that scores responses based on correctness."),
        HumanMessage(content=f"""
        Evaluate the following response against the reference answer. Assign a score between 0 and 1 based on correctness and provide a brief justification.

        Question: {question}
        Response: {model_answer}
        Reference Answer: {reference_answer}

        Score (0 to 1):
        Justification:
        """)
    ]
    response = local_llm(messages=messages)  # Using local_llm for evaluation
    evaluation_text = response.content.strip()
    
    try:
        score_line = evaluation_text.split('\n')[0]
        score = float(score_line.split(':')[1].strip())
        evaluation_scores.append(score)
    except Exception as e:
        print(f"Error parsing score: {e}")
        evaluation_scores.append(0.0)

    average_score = sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0
    print(f'Average Correctness Score: {average_score:.2f}')
    return average_score

def evaluate_chunking_techniques(df, openai_api_key):
    # chunking_methods = ["basic"]
    chunking_methods = ["recursive", "basic", "character", "semantic"]
    #chunking_methods = ["character", "recursive", "semantic"]
    results = []

    for method in chunking_methods:
        print(f"Evaluating chunking method: {method}")

        for i, row in df.iterrows():
            download_dir = "pdf_documents"
            os.makedirs(download_dir, exist_ok=True)
            pdf_url = row['doc_link']
            doc_name = row['doc_name']
            question = row['question']
            ref_answer = row['answer']
            ref_context = row['evidence_text']

            doc_path = os.path.join(download_dir, f"{doc_name}.pdf")
            #save_path = f"downloads/{row['financebench_id']}.pdf"
            download_pdf(pdf_url, doc_path)
            print(i)
            print(doc_name)
            print(pdf_url)
            if os.path.isfile(doc_path):
                text = extract_text_from_pdf(doc_path)
                documents = chunk_text(text, method=method)
                vectors = create_embeddings(documents, openai_api_key)
                
                store_embeddings_in_chroma(documents, vectors, collection_name=f"Finance_bench_{method}")

                model_answer = rag(question, collection_name=f"Finance_bench_{method}", api_key=openai_api_key)

                cosine_similarity_score = calculate_cosine_similarity(model_answer, ref_context)
                bert_score_value = calculate_bertscore(model_answer, ref_context)
                llm_eval = evaluate_llm_responses(question, model_answer, ref_context)

                results.append({
                    "doc_name" : doc_name,
                    "method": method,
                    "question": question,
                    "ref_answer": ref_answer,
                    "model_answer": model_answer,
                    "cosine_similarity": cosine_similarity_score,
                    "bert_score": bert_score_value,
                    "llm_eval": llm_eval
                })

    return pd.DataFrame(results)

import csv

def main():
    df = pd.read_csv("filtered_PatronousAIfinancebench.csv")
    download_dir = "documents_QE"
    if not(os.path.exists(download_dir)):
        os.makedirs(download_dir, exist_ok=True)
    return df


    
test_data = main()
results_df = evaluate_chunking_techniques(test_data, openai_api_key = "")

#Save results to a CSV file
results_df['model_answer'] = results_df['model_answer'].str.replace('\n', '<newline>').replace(',', '<comma>')
results_df['ref_answer'] = results_df['ref_answer'].str.replace('\n', '<newline>').replace(',', '<comma>')
results_df.to_csv("chunking_evaluation_results_basic.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

# Calculate metrics
def calculate_metrics(results_df):
    metrics = results_df.groupby('method').agg({
        'cosine_similarity': 'mean',
        'bert_score': 'mean',
        'llm_eval': 'mean'
    }).reset_index()
    return metrics

metrics = calculate_metrics(results_df)
print(metrics)

