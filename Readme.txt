Project Overview

** IF THE CODE IS TAKING TOO LONG TO RUN IT ON YOUR LOCAL MACHINE YOU CAN RUN THE CODE ON GOOGLE COLAB**
**FOR THE EXACT COMMANDS TO RUN THE CODE PLEASE GO THROUGH THE 'Instruction.txt' FILE**


This project is designed to facilitate efficient resume search and retrieval using FAISS (Facebook AI Similarity Search) and the HuggingFace model for embeddings. It includes both a backend (FastAPI) and a frontend (Streamlit) to allow users to query resumes and obtain relevant information.

Project Structure

The zip file contains the following key components:

1. Documentation

Vector Database Setup: Details the configuration process for the vector database using FAISS, including embedding storage and retrieval mechanisms.
API Documentation: Provides information on the API endpoints, how to interact with the backend, and sample queries with expected responses.
Report: Explains the approach to data handling, embedding generation, RAG (Retrieval-Augmented Generation) pipeline development, and API creation. It also discusses challenges encountered during the project and the solutions implemented.

2. Backend Code (app1.py)
The backend is built using FastAPI and handles the following tasks:

-Text extraction from PDFs and CSV files
-Embedding generation using HuggingFace models
-Vector search using FAISS for retrieving relevant resume segments
-Interacting with the inference API to generate answers to user queries


3. Frontend Code (streamlit_app.py)
The frontend is implemented using Streamlit and allows users to:
-Input questions related to resumes
-Connect with the backend to fetch and display relevant responses

4. requirements.txt
Contains all the necessary Python libraries required to run the project, such as FastAPI, Streamlit, PyPDF2, FAISS, HuggingFace libraries, and more.

5. .env
Stores the Hugging Face API token required for accessing the inference API.

6. Data
Resume Data (Data/): Contains PDF files of resumes organized by category.
Metadata (Resume.csv): A CSV file containing metadata, such as resume categories, IDs, and text extracted from resumes.

7. instruction.txt
Provides detailed setup instructions and guidance on how to run the project on both a local machine and Colab.



FOLDER STRUCTURE

/project_root
│
├── app1.py                 # Backend code (FastAPI)
├── streamlit_app.py       # Frontend code (Streamlit)
├── requirements.txt       # List of required packages
├── .env                   # Environment variables (Hugging Face API token)
├── readme.txt             # Detailed project documentation
├── instruction.txt        # Detailed setup and execution instructions
│
├── Documentation/
│   ├── VectorDatabaseSetup.pdf
│   ├── APIDocumentation.pdf
│   └── Report.pdf
│
├── Data/
│   ├── Category1/
│   │   ├── resume1.pdf
│   │   └── resume2.pdf
│   ├── Category2/
│   │   ├── resume3.pdf
│   │   └── resume4.pdf
│   └── ...
│
├── Resume.csv             # CSV file containing metadata of resumes
└── archive.zip            # Original data for upload to Colab




