import json
import os
import csv
from pathlib import Path

DEFAULT_SEARCH_LIMIT:int = 10 # number of search results provided
DEFAULT_CHUNK_SIZE: int = 200 # basic chunking chunk size by word count
DEFAULT_OVERLAP: int = 20 # basic chunking overlap by word count
DEFAULT_MAX_CHUNK_SIZE = 4 # semantic chunking max chunk size by number of sentences
DEFAULT_SEMANTIC_OVERLAP = 1 # semantic chunking overlap by number of sentences
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_PATH / "data" / "movies.json"
STOPWORD_PATH = PROJECT_PATH / "data" / "stopwords.txt"
CACHE_PATH = PROJECT_PATH / "cache"
MOVIE_EMBEDDINGS_PATH = PROJECT_PATH / "cache" / "movie_embeddings.npy"
MOVIE_EMBEDDINGS_MAP_PATH = PROJECT_PATH / "cache" / "movie_embeddings_map.pkl"
CHUNK_EMBEDDINGS_PATH = PROJECT_PATH / "cache" / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = PROJECT_PATH / "cache" / "chunk_metadata.json"


def load_database():
    with open(DATA_PATH) as f:
        return json.load(f)["movies"] 
    
def load_stopwords():
    with open(STOPWORD_PATH, "r") as f:
        stop = f.read().splitlines()
    return stop