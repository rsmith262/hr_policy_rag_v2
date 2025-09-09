# ingest/create_index.py
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

load_dotenv()
endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
key = os.environ["AZURE_SEARCH_KEY"]
index_name = os.environ["AZURE_SEARCH_INDEX"]

# Set to your embedding model's dimension (e.g., 3072 for text-embedding-3-large)
EMBED_DIM = 3072

def create_index():
    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="url", type=SearchFieldDataType.String),
        # IMPORTANT: use vector_search_profile_name (Python) -> maps to vectorSearchProfile (REST)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-alg")],
        profiles=[VectorSearchProfile(name="vector-profile",
                                      algorithm_configuration_name="hnsw-alg")],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

    # Dev convenience: drop/recreate
    try:
        client.delete_index(index_name)
    except Exception:
        pass

    client.create_index(index)
    print(f"Index '{index_name}' created.")

if __name__ == "__main__":
    create_index()



