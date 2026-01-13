import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    CACHE_PATH,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_EMBEDDINGS_MAP_PATH,
    load_database,
    DEFAULT_SEARCH_LIMIT,
    )

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_path = CACHE_PATH
        self.movie_embeddings_path = MOVIE_EMBEDDINGS_PATH
        self.movie_embeddings_map_path = MOVIE_EMBEDDINGS_MAP_PATH
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
    
    # Search function comparing query to documents
    def search(self, query, limit: int | None = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings, loaded. Call 'load_or_create_embeddings' function first.")
        else:
            embedded_query = self.generate_embedding(query)
            cosine_similarities = {i: cosine_similarity(embedded_query, vec2) for i, vec2 in enumerate(self.embeddings)}
            cosine_similarities = {k: v for k, v in sorted(cosine_similarities.items(), key = lambda item: item[1], reverse=True)}
            
            top_results = []
            counter = 0 # used to break when limit is reached
            for embed_id, score in cosine_similarities.items():
                doc_id = self.embeddings_map.get(embed_id)
                title = self.document_map.get(doc_id).get('title')
                description = self.document_map.get(doc_id).get('description')
                top_results.append({'score': score, 'title': title, 'description': description})
                counter +=1
                if counter >= limit:
                    break
            return top_results

    # Generate an embedding for an individual string
    def generate_embedding(self, text):
        # Raise error if no text of if only white space
        if not text or not text.strip():
            raise ValueError("Input text is empty or only whitespace - cannot generate embedding.")
        else:
            # The encode function takes a list of strings as input
            items_to_embed = [text]
            embeddings = self.model.encode(items_to_embed)
            first_item = embeddings[0]
            return first_item
    
    def build_document_map(self, documents):
        self.documents = documents
        #self.document_map = {doc.get('id'): f"{doc.get('title')}: {doc.get('description')}" for doc in self.documents}
        self.document_map = {doc.get('id'): {'title': doc.get('title'), 'description': doc.get('description')} for doc in self.documents}

    def build_embeddings(self, documents):
        # Ensure that document map is populated
        self.build_document_map(documents)
        # Create mapping of embedding to document key {embedding position: document_key}
        self.embeddings_map = {i: k for i, k in enumerate(self.document_map)}
        # Generate and export embeddings (and embeddings map) to cache
        movie_strings = [f"{info.get('title')}: {info.get('description')}" for id, info in self.document_map.items()]
        self.embeddings = self.model.encode(sentences=movie_strings, show_progress_bar=True)
        self.save_embeddings()
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        # Load embeddings (and embeddings_map) if in cache & populate variables initialized to None
        try:
            self.open_embeddings()
            self.build_document_map(documents)
            # Confirm that the number of entries is consistent, otherwise raise error to go to except block
            if self.embeddings is not None and len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                raise ValueError
        # Generate new embeddings cache if try block fails
        except:
            self.build_embeddings(documents)
            return self.embeddings
    
    def save_embeddings(self):
        if self.embeddings is None:
            raise ValueError
        if not os.path.isdir(self.cache_path):
            os.mkdir(self.cache_path)
        with open(self.movie_embeddings_path, 'wb') as f:
            np.save(f, self.embeddings)
            print("Embeddings saved.")
        with open(self.movie_embeddings_map_path, 'wb') as f:
            pickle.dump(self.embeddings_map, f)
            print("Embeddings map saved.")
    
    def open_embeddings(self):
        if not os.path.isfile(self.movie_embeddings_path):
            raise FileNotFoundError
        if not os.path.isfile(self.movie_embeddings_map_path):
            raise FileNotFoundError
        with open(self.movie_embeddings_path, 'rb') as f:
            self.embeddings = np.load(f)
            print("Embeddings loaded from cache.")
        with open(self.movie_embeddings_map_path, 'rb') as f:
            self.embeddings_map = pickle.load(f)
            print("Embeddings map loaded from cache.")

    
def verify_model():
    m = SemanticSearch() # initialize the model
    print(f'Model loaded: {m.model}')
    print(f'Max sequence length: {m.model.max_seq_length}')

def embed_text(text):
    m = SemanticSearch() # initialize the model
    embedding = m.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    m = SemanticSearch()
    documents = load_database()
    m.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {m.embeddings.shape[0]} vectors in {m.embeddings.shape[1]} dimensions")

def embed_query_text(query):
    m = SemanticSearch()
    embedded_query = m.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedded_query[:5]}")
    print(f"Shape: {embedded_query.shape}")
    return embedded_query

def search_command(query, limit):
    m = SemanticSearch()
    documents = load_database()
    m.load_or_create_embeddings(documents)
    results = m.search(query, limit)
    return results

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # Treat division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Calculate cosine similarity
    else:
        return dot_product / (norm1 * norm2)

# Test code
#verify_embeddings()
#documents = load_database()
#m = SemanticSearch()
#embeddings = m.build_embeddings(documents[:10])
#embeddings = m.build_embeddings(documents)
#m.save_embeddings()
#m.open_embeddings()
#m.load_or_create_embeddings(documents)
#query = "funny bear movies"
#results = m.search(query)
#for item in results:
#    print(item.get('title'))
#print(results)
#print(m.embeddings)
