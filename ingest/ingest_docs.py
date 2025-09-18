import os, uuid, tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from typing import List, Tuple
from azure.storage.blob import BlobServiceClient

load_dotenv()

endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
key = os.environ["AZURE_SEARCH_KEY"]
index_name = os.environ["AZURE_SEARCH_INDEX"]

emb = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEP"],
    api_version=os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2024-10-01-preview"),
)

def iterate_local_pdfs(docs_root: str) -> List[Tuple[str, str, str]]:
    acc = os.environ.get("BLOB_ACCOUNT_NAME", "")
    container = os.environ.get("BLOB_CONTAINER", "docs")
    files = []
    for root, _, fnames in os.walk(docs_root):
        for fn in fnames:
            if fn.lower().endswith(".pdf"):
                local_path = os.path.join(root, fn)
                rel = os.path.relpath(local_path, start=docs_root).replace("\\", "/")
                bare_url = f"https://{acc}.blob.core.windows.net/{container}/{rel}" if acc else None
                files.append((f"{container}/{rel}", bare_url, local_path))
    return files

def iterate_blob_pdfs() -> List[Tuple[str, str, str]]:
    conn = os.environ.get("AZURE_BLOB_CONN")
    container = os.environ.get("BLOB_CONTAINER", "docs")
    if not conn:
        return []
    bsc = BlobServiceClient.from_connection_string(conn)
    cont = bsc.get_container_client(container)
    out = []
    for blob in cont.list_blobs():
        if blob.name.lower().endswith(".pdf"):
            bare = f"https://{bsc.account_name}.blob.core.windows.net/{container}/{blob.name}"
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                cont.download_blob(blob.name).readinto(tmp)
                out.append((f"{container}/{blob.name}", bare, tmp.name))
    return out

def load_documents():
    items = iterate_blob_pdfs()
    return items if items else iterate_local_pdfs("docs")

def chunk_and_prepare():
    items = load_documents()
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for blob_path, bare_url, local_path in items:
        docs = PyPDFLoader(local_path).load()
        chunks = splitter.split_documents(docs)
        for i, d in enumerate(chunks):
            d.metadata["source"] = blob_path
            d.metadata["url"] = bare_url
            if "page" not in d.metadata and "page_number" in d.metadata:
                d.metadata["page"] = d.metadata["page_number"]
            d.metadata["chunk_id"] = f"{i:06d}"
        all_chunks.extend(chunks)
    return all_chunks

def to_search_docs(chunks):
    texts = [c.page_content for c in chunks]
    vectors = emb.embed_documents(texts)
    payload = []
    for c, vec in zip(chunks, vectors):
        url = c.metadata.get("url")
        if url and c.metadata.get("page") and str(url).lower().endswith(".pdf"):
            url = f"{url}#page={int(c.metadata['page'])}"
        payload.append({
            "id": str(uuid.uuid4()),
            "content": c.page_content,
            "content_vector": vec,
            "source": c.metadata.get("source", "blob"),
            "page": int(c.metadata["page"]) if "page" in c.metadata else None,
            "chunk_id": c.metadata.get("chunk_id"),
            "url": url,
        })
    return payload

def upload(payload):
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    for i in range(0, len(payload), 1000):
        batch = payload[i:i+1000]
        r = client.upload_documents(documents=batch)
        fails = [x for x in r if not x.succeeded]
        if fails: print("Failed:", fails)
    print(f"Uploaded {len(payload)} docs.")

if __name__ == "__main__":
    payload = to_search_docs(chunk_and_prepare())
    upload(payload)
