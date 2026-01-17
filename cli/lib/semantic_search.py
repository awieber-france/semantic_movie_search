import numpy as np
import os
import pickle
import re
import json
from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    CACHE_PATH,
    MOVIE_EMBEDDINGS_PATH,
    MOVIE_EMBEDDINGS_MAP_PATH,
    load_database,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEMANTIC_OVERLAP,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    )

###------------SEMANTIC SEARCH METHODS (BASIC AND ADVANCED)------------###

# Basic semantic search where title + description are converted to a single embedding
class SemanticSearch():
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.cache_path = CACHE_PATH
        self.movie_embeddings_path = MOVIE_EMBEDDINGS_PATH
        self.movie_embeddings_map_path = MOVIE_EMBEDDINGS_MAP_PATH
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
    
    # Basic search function comparing query to documents (title + description) - the entire document becomes one embedding (same for query)
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

    # Split strings into chunks (by word count)
    def chunk(self, text, chunk_size: int | None = DEFAULT_CHUNK_SIZE, overlap: int | None = DEFAULT_OVERLAP):
        if overlap >= chunk_size:
            raise ValueError(f"The overlap (value supplied = {overlap}) must be strictly smaller than the chunk size (value supplied = {chunk_size})!")
        if overlap < 0:
            raise ValueError(f"The overlap must be a positive integer (value supplied = {overlap})!")
        words = str.rsplit(text) # use default whitespace separator (any whitespace!)
        i = 0 # used to position index using chunk_size and overlap
        j = 0 # used to detect when at end of text
        chunks = []
        while j < len(words):
            chunks.append(' '.join(words[i:i+chunk_size]))
            j = i + chunk_size
            i = i + chunk_size - overlap
        return chunks

    # Chunk by sentences, keeping semantic context
    def semantic_chunk(self, text, max_chunk_size: int | None=DEFAULT_MAX_CHUNK_SIZE, overlap: int | None = DEFAULT_SEMANTIC_OVERLAP):
        if overlap >= max_chunk_size:
            raise ValueError(f"The overlap (value supplied = {overlap}) must be strictly smaller than the max chunk size (value supplied = {max_chunk_size})!")
        if overlap < 0:
            raise ValueError(f"The overlap must be a positive integer (value supplied = {overlap})!")
        # Split into sentences using regex rules
        sentences = regex_split(text)
        # Make chunks of one or more sentences, with overlap
        chunks = []
        i = 0
        j = i + max_chunk_size
        if len(sentences) == 0:
            return chunks
        # If en index is at or beyond final sentence, then mark as last cycle
        if j >= len(sentences):
                last_cycle = True
        else:
            last_cycle = False
        while i < len(sentences):
            chunks.append(' '.join(sentences[i:j]))
            # Break if this was the last cycle
            if last_cycle:
                break
            # Prepare next chunk indices
            j = j + max_chunk_size - overlap
            i = j - max_chunk_size
            # If next index is at or exceed end, then mark as last cycle
            if j >= len(sentences):
                last_cycle = True
        return chunks

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

# Advanced semantic search using chunk embeddings (from movie descriptions only) - query remains a single embedding
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = 'all-MiniLM-L6-v2') -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = CHUNK_EMBEDDINGS_PATH
        self.chunk_metadata_path = CHUNK_METADATA_PATH
    # Advanced semantic query using chunks
    def search_chunks(self, query: str, limit: int | None = DEFAULT_SEARCH_LIMIT):
        # Generate a single embedding for the query (not chunked)
        query_embedding = self.generate_embedding(query)
        # Run internal functions to get scores and create results
        chunk_scores = self._score_chunks(query_embedding)
        document_scores = self._map_chunk_document_scores(chunk_scores)
        top_results = self._best_chunk_document_matches(document_scores, limit)
        return top_results
    # Calculate scores for each chunk and store in dict: chunk_idx, movie_idx, score
    def _score_chunks(self, query_embedding):
        chunk_scores = []
        for embed, meta in zip(self.chunk_embeddings, self.chunk_metadata.get('chunks')):
            score = cosine_similarity(query_embedding, embed)
            chunk_scores.append({'chunk_idx': meta.get('chunk_idx'), 'movie_idx': meta.get('movie_idx'), 'score': score})
        return chunk_scores
    # Create dictionary of movies with their highest score
    def _map_chunk_document_scores(self, chunk_scores):
        document_scores = dict()
        for chunk in chunk_scores:
            movie_idx = chunk.get('movie_idx')
            score = chunk.get('score')
            if document_scores.get(movie_idx) is None:
                document_scores[movie_idx] = score
            elif score > document_scores.get(movie_idx):
                document_scores[movie_idx] = score
            else:
                continue
        return document_scores
    # Rank movies by score and provide the data, within the print limit of results
    def _best_chunk_document_matches(self, document_scores, limit: int):
        if limit < 0:
            raise ValueError('The value provided for the maximum number of results is negative. It must be a positive integer.')
        ordered_matches = {k: v for k, v in sorted(document_scores.items(), key = lambda item: item[1], reverse=True)}
        results = []
        counter = 0
        for id, score in ordered_matches.items():
            results.append({'id': id,
                            'title': self.document_map.get(id).get('title'),
                            'description': self.document_map.get(id).get('description'),
                            'score': score,
                            'metadata': self.document_map.get(id).get('metadata')})
            counter += 1
            # Exit early when results limit is reached
            if counter > limit:
                return results
        return results
    # Split each movie's description into chunks and embed (title is not included)
    def build_chunk_embeddings(self, documents):
        # Create documents and document_map objects
        self.build_document_map(documents)
        # Initalize chunk lists
        self.chunk_embeddings = []
        self.chunk_metadata = []
        # Build chunk embeddings and save chunk metadata
        all_chunks = []
        for id, info in self.document_map.items():
            title = info.get('title')
            desc = info.get('description')
            # Make movie string to send chunk (exclued description if empty)
            if not desc or not desc.strip():
                continue
                #movie_string = title
            else:
                # Remove white space before/after
                movie_string = desc.strip()
                #movie_string = f"{title}: {desc}"
            chunks = self.semantic_chunk(movie_string)
            all_chunks.extend(chunks)
            self.chunk_metadata.extend([{'movie_idx': id, 'chunk_idx': chunk_id, 'total_chunks': len(chunks)} for chunk_id in range(len(chunks))])
        # Encode embeddings
        self.chunk_embeddings = self.model.encode(sentences=all_chunks, show_progress_bar=True)
        # Save embeddings and chunk metadata
        self.save_chunk_embeddings()
        return self.chunk_embeddings
    # Load from cache or build chunk embeddings for documents
    def load_or_create_chunk_embeddings(self, documents):
        # Load embeddings (and embeddings_map) if in cache & populate variables initialized to None
        try:
            self.open_chunk_embeddings()
            self.build_document_map(documents)
            # Confirm that the number of entries is consistent, otherwise raise error to go to except block
            if self.chunk_embeddings is not None and len(self.chunk_embeddings) == len(self.chunk_metadata.get('chunks')) and self.chunk_metadata.get('total_chunks')== len(self.chunk_metadata.get('chunks')):
                return self.chunk_embeddings
            else:
                raise ValueError
        # Generate new embeddings cache if try block fails
        except:
            self.build_chunk_embeddings(documents)
            return self.chunk_embeddings
    # Save chunk embeddings and metadata to cache
    def save_chunk_embeddings(self):
        if self.chunk_embeddings is None:
            raise ValueError
        if not os.path.isdir(self.cache_path):
            os.mkdir(self.cache_path)
        with open(self.chunk_embeddings_path, 'wb') as f:
            np.save(f, self.chunk_embeddings)
            print("Chunk embeddings saved.")
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(self.chunk_embeddings)}, f, indent=2)
            print("Chunk embeddings metadata saved.")
    # Open chunk embeddings and metadata from cache
    def open_chunk_embeddings(self):
        if not os.path.isfile(self.chunk_embeddings_path):
            raise FileNotFoundError
        if not os.path.isfile(self.chunk_metadata_path):
            raise FileNotFoundError
        with open(self.chunk_embeddings_path, 'rb') as f:
            self.chunk_embeddings = np.load(f)
            print("Chunk embeddings loaded from cache.")
        with open(self.chunk_metadata_path, 'r') as f:
            self.chunk_metadata = json.load(f)
            print("Chunk embeddings metadata loaded from cache.")
        

###------------TOOLS------------###
#  Scoring of search results
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
    
# Regex splitting (and begin/end whitespace removal) of a string
def regex_split(text, pattern: str | re.Pattern | None = r"(?<=[.!?])\s+"):
    #pattern=r"(?<=[.!?])(?!\d)\s+" (decimal treatment)
    # If None is passed, return text in list
    if pattern is None:
        return [text.strip()]
    split_text = re.split(pattern=pattern, string=text)
    split_text = [string.strip() for string in split_text]
    split_text = [string for string in split_text if len(string)>0]
    return split_text

###------------COMMANDS TO EXPLOIT SemanticSearch CLASS------------###
# Command for use of SemanticSearch class: Check what model and parameters are used
def verify_model():
    m = SemanticSearch() # initialize the model
    print(f'Model loaded: {m.model}')
    print(f'Max sequence length: {m.model.max_seq_length}')

# Command for use of SemanticSearch class: simple string embedding command (for debug and information)
def embed_text(text):
    m = SemanticSearch() # initialize the model
    embedding = m.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

# Command for use of SemanticSearch class: load or create embeddings and print info
def verify_embeddings():
    m = SemanticSearch() # initialize the model
    documents = load_database()
    m.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {m.embeddings.shape[0]} vectors in {m.embeddings.shape[1]} dimensions")

# Command for use of SemanticSearch class: take a query, embed it, and provide embedding characteristics (for debug and information)
def embed_query_text(query):
    m = SemanticSearch() # initialize the model
    embedded_query = m.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedded_query[:5]}")
    print(f"Shape: {embedded_query.shape}")
    return embedded_query

# Command for use of SemanticSearch class: semantic search command
def search_command(query, limit):
    m = SemanticSearch() # initialize the model
    documents = load_database()
    m.load_or_create_embeddings(documents)
    results = m.search(query, limit)
    return results

def chunk_command(query, chunk_size, overlap):
    m = SemanticSearch() # initialize the model
    chunks = m.chunk(query, chunk_size, overlap)
    print(f'Chunking {len(query)} characters.')
    for i, chunk in enumerate(chunks):
        print(f'{i+1}. {chunk}')

def semantic_chunk_command(query, max_chunk_size, overlap):
    m = SemanticSearch() # initialize the model
    chunks = m.semantic_chunk(query, max_chunk_size, overlap)
    print(f'Semantically chunking {len(query)} characters.')
    for i, chunk in enumerate(chunks):
        print(f'{i+1}. {chunk}')

# Load from cache or build embeddings of documents in chunks
def embed_chunks_command():
    documents = load_database()
    m = ChunkedSemanticSearch() # initialize the model
    chunk_embeddings = m.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")

def search_chunked_command(query, limit):
    documents = load_database()
    m = ChunkedSemanticSearch() # initialize the model
    m.load_or_create_chunk_embeddings(documents) # load or generate embeddings of documents
    top_results = m.search_chunks(query, limit)
    # print results for each document (limit description to 100 characters)
    for i, result in enumerate(top_results):
        print(f"\n{i+1}. {result.get('title')} (score: {result.get('score'):.4f})")
        print(f"   {result.get('description')[:100]}...")

###------------MANUAL TESTING------------###
#NOTA: remove ".lib" from "from lib.search_utils import"
"""
# Build new basic cache
documents = load_database()
m = SemanticSearch()
m.build_embeddings(documents)

# Build new chunked cache
documents = load_database()
m = ChunkedSemanticSearch()
m.build_chunk_embeddings(documents)

# Run queries
query='superhero action movie'
search_command(query, limit=10)
search_chunked_command(query, limit=10)
"""