import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command
from lib.semantic_search import chunk_command, semantic_chunk_command, embed_chunks_command, search_chunked_command
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEMANTIC_OVERLAP,
    )

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # Verify parser
    verify_parser = subparsers.add_parser("verify", help="Verify semantic model.")
    # Embed text parser
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a single string of text.")
    embed_text_parser.add_argument("text", type=str, help="Text string")
    # Verify embeddings parser
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings and ensure their conformity in the cache. Build cache if missing.")
    # Embed query parser
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query.")
    embed_query_parser.add_argument("query", type=str, help="Query string to embed")
    # Semantic search parser
    search_parser = subparsers.add_parser("search", help="Semantic search query.")
    search_parser.add_argument("query", type=str, help="The query to submit with the search funciton.")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results to return.")
    # Chunk parser
    chunk_parser = subparsers.add_parser("chunk", help="Basic approach for splitting a string into chunks by number of words.")
    chunk_parser.add_argument("text", type=str, help="String of text to split into chunks, using whitespaces to count words.")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of words per chunk, except last chunk whiche may be smaller.")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap between chunks in number of words. Typically 20 percent.")
    # Semantic chunk parser
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk by sentences, keeping semantic meaning.")
    semantic_chunk_parser.add_argument("text", type=str, help="String of text to split into chunks by sentences.")
    semantic_chunk_parser.add_argument("--max-chunk-size", type = int, default=DEFAULT_MAX_CHUNK_SIZE, help="Maximum number of sentences contained in a chunk.")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_SEMANTIC_OVERLAP, help="Overlap between chunks in number of sentences.")
    # Load or build chunk embeddings parser
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Load from cache or build embeddings of documents in chunks")
    # Search chunk embeddings parser
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Advanced semantic search query of documents (using chunking).")
    search_chunked_parser.add_argument("query", type=str, help="The query to submit with the search funciton.")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results to return.")

    args = parser.parse_args()

    match args.command:
        case 'verify':
            # Check imported model
            verify_model()
        case 'embed_text':
            # Embed a single text string
            embed_text(args.text)
        case 'verify_embeddings':
            verify_embeddings()
        case 'embedquery':
            embed_query_text(args.query)
        case 'search':
            results = search_command(args.query, args.limit)
            for i, item in enumerate(results):
                print(f"{i+1}. {item.get('title')} (score: {'{0:.4f}'.format(item.get('score'))})")
                print(f"   {item.get('description')[:100]}...")
                # leave a space between results
                if i+1 < len(results):
                    print()
        case 'chunk':
            chunk_command(args.text, args.chunk_size, args.overlap)
        case 'semantic_chunk':
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case 'embed_chunks':
            embed_chunks_command()
        case 'search_chunked':
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()