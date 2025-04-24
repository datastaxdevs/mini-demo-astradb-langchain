"""
Required dependencies:

    pip install \
        "langchain>=0.3,<0.4" \
        "langchain-astradb>=0.6,<0.7" \
        "langchain-openai>=0.3,<0.4"

Requires a `.env` file with environment variables, see `template.env`.
"""


# Import dependencies
import os
import requests
from getpass import getpass

from astrapy.info import VectorServiceOptions
from langchain_astradb import AstraDBVectorStore

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


# Load environment variables
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE") or None
ASTRA_DB_API_KEY_NAME = os.environ.get("ASTRA_DB_API_KEY_NAME") or None

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or None


# Create a vector store
vectorize_options = VectorServiceOptions(
    provider="openai",  # Change these if using another embedding provider/model
    model_name="text-embedding-3-small",
    authentication={"providerKey": ASTRA_DB_API_KEY_NAME},
)
vector_store = AstraDBVectorStore(
    collection_name="langchain_integration_demo_vectorize",
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=ASTRA_DB_KEYSPACE,
    collection_vector_service_options=vectorize_options,
)


# Load data
philo_dataset = requests.get(
    "https://raw.githubusercontent.com/"
    "datastaxdevs/mini-demo-astradb-langchain/"
    "refs/heads/main/data/philosopher-quotes.json"
).json()

print("An example entry:")
print(philo_dataset[16])


# Process dataset
documents_to_insert = []

for entry_idx, entry in enumerate(philo_dataset):
    metadata = {
        "author": entry["author"],
        **entry["metadata"],
    }
    # Construct the Document, with the quote and metadata tags
    new_document = Document(
        id=entry["_id"],
        page_content=entry["quote"],
        metadata=metadata,
    )
    documents_to_insert.append(new_document)

print(f"Ready to insert {len(documents_to_insert)} documents.")
print(f"Example document: {documents_to_insert[16]}")


# Insert documents
inserted_ids = vector_store.add_documents(documents_to_insert)

print(f"\nInserted {len(inserted_ids)} documents: {', '.join(inserted_ids[:3])} ...")


# Verify the integration
results = vector_store.similarity_search("Our life is what we make of it", k=3)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
