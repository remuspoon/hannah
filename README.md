# ChromaDB setup for Reddit mental health posts

This folder builds a vector database of r/mentalhealth submissions and uploads them to ChromaDB (cloud) for semantic search.

## How it works

1. **Load data** — Reads `data/mentalhealth_submissions.jsonl` (Reddit submission dump, one JSON object per line).
2. **Filter** — Keeps only the last 2 years; drops `[deleted]`/`[removed]` posts and authors; keeps only posts with 100+ words in the body.
3. **Truncate** — ChromaDB has a 16KB document limit. Any post where `title + " | " + selftext` exceeds 16,000 characters has its body truncated so the full document fits.
4. **Embed & upload** — Each document is `title | selftext`. They’re embedded with OpenAI `text-embedding-3-small` and stored in a ChromaDB collection with metadata (author, permalink, subreddit, etc.). Upload runs in batches with progress saved so you can resume after errors.

Outputs: `data/clean_df.csv` (cleaned dataframe), `data/chroma_upload_progress.txt` (last uploaded batch index).

## Setup

### 1. Dependencies

From the project root (or this folder), use a venv and install:

```bash
pip install -r requirements.txt
```

### 2. Data

Put the Reddit submissions dump at:

```
chroma-db/data/mentalhealth_submissions.jsonl
```

Each line must be a single JSON object with at least: `title`, `selftext`, `author`, `created_utc`, `permalink`, `id`, `subreddit`, `subreddit_id`, `link_flair_text`, `retrieved_on`.

### 3. Environment variables

Create a `.env` in the project root (or where `dotenv.load_dotenv()` is run) with:

- **`CHROMA_API_KEY`** — Your ChromaDB cloud API key (required for upload).
- **`OPENAI_API_KEY`** — Used by the ChromaDB collection’s embedding function (OpenAI).

### 4. ChromaDB collection

The notebook uses an existing cloud collection named `reddit-bot-vdb`. Create it in the ChromaDB dashboard (or uncomment the “Create ChromaDB” cell in the notebook to create it with the OpenAI embedding function).

### 5. Run the notebook

Open `setup.ipynb` and run all cells. For the batch upload, if it stops (e.g. network/API error), run the upload cell again; it will resume from the index stored in `data/chroma_upload_progress.txt`.
