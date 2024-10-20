from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient
import re
import torch
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ignore FutureWarnings related to clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
# Check CUDA availability and set the device
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {device}")
else:
    device = "cpu"
    print(f"Using device: {device}")

app = FastAPI()

class UserInput(BaseModel):
    question: str

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to extract resumes from CSV and PDFs
def extract_resumes_from_csv_and_pdfs(csv_file_path, pdf_folder_path):
    df = pd.read_csv(csv_file_path)
    resume_data = []

    for _, row in df.iterrows():
        resume_text = row['Resume_str']
        category = row['Category']

        if pd.isna(resume_text) or len(resume_text.strip()) == 0:
            pdf_path = os.path.join(pdf_folder_path, category, f"{row['ID']}.pdf")
            if os.path.exists(pdf_path):
                resume_text = extract_text_from_pdf(pdf_path)

        resume_data.append((resume_text, category))
    return resume_data

# Load resumes
csv_file_path = r"C:\Users\yash\Downloads\Rag_Yash\Resume.csv"  # Adjust this path as needed
pdf_folder_path = r"C:\Users\yash\Downloads\Rag_Yash\data"  # Adjust this path as needed
resume_data = extract_resumes_from_csv_and_pdfs(csv_file_path, pdf_folder_path)
df = pd.DataFrame(resume_data, columns=['Resume_text', 'Category'])

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

df['Processed_text'] = df['Resume_text'].apply(preprocess_text)

# Initialize embeddings and index
embedder = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
embedding_list = embedder.embed_documents(df['Processed_text'].tolist())

# Print generated embeddings for inspection
print(f"Generated embeddings shape: {len(embedding_list)}")

embedding_matrix = np.array(embedding_list).astype(np.float32)
embedding_matrix = np.ascontiguousarray(embedding_matrix)

# Initialize FAISS index and ensure it's using GPU
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # Create the FAISS index (flat L2 index)
gpu_res = faiss.StandardGpuResources()

if torch.cuda.is_available():
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)  # Move to GPU
    print("FAISS index moved to GPU.")
else:
    gpu_index = index
    print("FAISS is using the CPU.")

# Add embeddings to the FAISS index
gpu_index.add(embedding_matrix)
print(f"FAISS index contains {gpu_index.ntotal} vectors.")

# Initialize BM25 for textual search
bm25 = BM25Okapi([text.split() for text in df['Processed_text']])

def handle_user_input(user_question):
    question_embedding = embedder.embed_query(user_question)
    question_embedding = np.array(question_embedding).astype(np.float32)
    question_embedding = np.ascontiguousarray(question_embedding)

    # FAISS search
    distances, indices = gpu_index.search(question_embedding.reshape(1, -1), k=5)
    faiss_results = df.iloc[indices[0]]['Processed_text'].tolist()

    # BM25 search
    bm25_results = bm25.get_top_n(user_question.split(), df['Processed_text'].tolist(), n=5)

    # Combine FAISS and BM25 results
    combined_results = faiss_results + bm25_results
    unique_results = list(set(combined_results))
    
    # Rank results by frequency of appearance
    ranked_results = sorted(unique_results, key=lambda x: combined_results.count(x), reverse=True)

    response = generate_response_from_inference_api(user_question, ranked_results)
    return response

# Update the inference API function to check token
def generate_response_from_inference_api(user_question, relevant_texts):
    sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    prompt = f"""
    You are an assistant that provides detailed answers based on the content of multiple resumes.
    
    Here are the relevant texts extracted from the resumes:
    {relevant_texts}
    
    The user has asked the following question:
    {user_question}
    
    Provide a detailed answer based on the information from all the resumes.
    """

    client = InferenceClient("mistralai/Mistral-Nemo-Instruct-2407", token=sec_key)
    response = client.text_generation(prompt, max_new_tokens=5000)
    return response


@app.post("/query/")
def handle_query(user_input: UserInput):
    question = user_input.question
    response = handle_user_input(question)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)