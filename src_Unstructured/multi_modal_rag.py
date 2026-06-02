import os
import json
from typing import List


from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
# Because i am on MacOS13 so using API
from langchain_unstructured import UnstructuredLoader

load_dotenv()

loader = UnstructuredLoader(
    file_path="docs/your-file.pdf",
    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    partition_via_api=True,
    chunking_strategy="by_title",  # chunks by document structure
    max_characters=1000,           # max chunk size
    combine_text_under_n_chars=200 # combines small chunks
)

documents = loader.load()



