# Semantic Search of Movie Database
## 1) Introduction
This code employs semantic search of a database of 5000 movies (or any database with identical structure). The *all-MiniLM-L6-v2* embeddings model from *sentece_transformers* is used to compare query vectors to combined title-description vectors.

Top 5 matches are supplied by default. Cosine similarity is used to rank the results.

## 2) Requirements
### Virtual environment
- Python package manager: uv 0.9.11
- Python version: ≥3.13

### Installation
- Perform a git pull of the project
- Navigate to the local git project folder in a terminal
- **curl -LsSf https://astral.sh/uv/0.9.11/install.sh | sh**
- **source $HOME/.local/bin/env**
- Call **uv build**

### Movie data:
Movie data must be downloaded separately and placed in the /data repository of the project. It can be downloaded at the following link:<br>
https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json

### Hardware and operating system:
This project should operate on any system capable of using the above uv and python versions. It was tested on macOS 13.7.8 with Intel Core i7 processor.

### Adapting to another database
- Use a json file having the same structure as the movie database.
- Adjust  **cli/lib/search_utils.py** as needed:
    - Modify the **load_database** function
    - Adjust the file name in the **DATA_PATH** variable

## 3) Using the files
- Navigate to the local git project folder in a terminal
- Build the cached database files for the first time (use later to verify):
    - **uv run cli/semantic_search_cli.py verify_embeddings**
- Type a command, for example:
    - **uv run cli/semantic_search_cli.py search "space adventure"** --limit 10
- Go to cli/lib/search_utils.py to adjust the default search limit.
- Call help function to get details of possible commands:
    - **uv run ./cli/semantic_search_cli.py --help**
    - **uv run cli/semantic_search_cli.py search --help**

## 4) References
This project was developped while following an online course from boot.dev for retrieval augmented generation:<br>
https://www.boot.dev/lessons/7a92d1c1-d202-481a-ae5f-14fc9f97b640