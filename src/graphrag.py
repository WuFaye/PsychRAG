import sys
import os
import logging
# import ollama
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from _llm import deepseek_response_if_catch
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)
import numpy as np
from time import time

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)
WORKING_DIR = "./psychrag_kg"

EMBED_MODEL = SentenceTransformer('BAAI/bge-large-zh-v1.5',  device = 'cuda',cache_folder='./psychrag_kg')

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=deepseek_response_if_catch,
        cheap_model_func=deepseek_response_if_catch,
        embedding_func=local_embedding,
    )

def kg_query(query_prompt: str, WORKING_DIR: str = None):
    
    result = rag.query(query_prompt, param=QueryParam(mode="global"))
    return result

def kg_insert(docu_text: str, WORKING_DIR: str = None):
    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    start = time()
    rag.insert(docu_text)
    print("indexing time:", time() - start)
