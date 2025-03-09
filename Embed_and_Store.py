import os
import torch
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from Utils import get_embedding, load_ppl_model, load_embed_model
from perplexity_chunking.chunk_rag import extract_by_html2text_db_nolist
from langchain.docstore.document import Document

def chunk_text_file(file_path, model, tokenizer, threshold, language='en'):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    chunks = extract_by_html2text_db_nolist(text, model, tokenizer, threshold, language)
    return chunks

def embed_single_doc(folder_path,ppl_model, ppl_tokenizer,embed_model,embed_tokenizer,device,batch_size = 8):
    persist_path = os.path.join(folder_path,"chroma_db")
    txt_folder_path = os.path.join(folder_path, "text")
    chroma_client = PersistentClient(path=persist_path)  #持久化客户端
    collection = chroma_client.get_or_create_collection(name="doc_embeddings")

    txt_files = sorted([f for f in os.listdir(txt_folder_path) if f.endswith(".txt")], key=lambda x: int(x.split(".")[0]))
    documents = []

    for filename in txt_files:
        page_id = int(filename.split(".txt")[0])
        file_path = os.path.join(txt_folder_path, filename)
        chunks = chunk_text_file(file_path, ppl_model, ppl_tokenizer, threshold = 0.5, language='en')
        print(f'Page {page_id} has {len(chunks)} chunks')
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"page": page_id}))

    #批量embed
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch_contents = [doc.page_content for doc in documents[i:i + batch_size]]
        batch_embeddings = get_embedding(batch_contents, embed_model,embed_tokenizer ,device, return_tensor=False)
        embeddings.extend(batch_embeddings)

    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(i)],
            embeddings=embeddings[i],
            documents=[doc.page_content],
            metadatas=[{"page": doc.metadata["page"]}]
        )
    return

def embed_all_docs(dir, model_name = 'Qwen/Qwen2.5-1.5B-Instruct'):
    ppl_model, ppl_tokenizer,device = load_ppl_model(model_name)
    embed_model,embed_tokenizer,device = load_embed_model(model_name = "BAAI/bge-m3")
    docs = os.listdir(dir)
    for doc in docs:
        doc_dir = os.path.join(dir, doc)
        embed_single_doc(doc_dir,ppl_model, ppl_tokenizer,embed_model,embed_tokenizer,device)


embed_all_docs('OCR_DOC')



