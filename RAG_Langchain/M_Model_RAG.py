import os
os.environ["TESSERACT_CMD"] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#unstructred fro document parsing
import json
from typing import List

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()




def partition_document(file_path: str):
    print(f"Document:{file_path}")

    elements = partition_pdf(
        filename=file_path, #file Path
        strategy="hi_res", #most accurate (but slower) processing method of extraction
        infer_table_structure=True, #for tables , not stored as jumbled text
        extract_image_block_types=['Image'], # for image
        extract_image_block_to_payload=True,  # store image  as base64
        languages=["English"]   
        )

    print(f"Extracted {len(elements)} elements")
    return elements

file_path = "data/New.pdf"
elements = partition_document(file_path)

