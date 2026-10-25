# Hearing Aid Technical Documentation RAG 

`rag_eval.py` builds a retrieval-augmented Q&A system over public Starkey Omega AI documentation and evaluates it against a prompting-only baseline. PDFs are chunked and embedded locally with `BAAI/bge-small-en-v1.5`, stored in Chroma, and answered by an Azure AI Foundry chat deployment (`gpt-6-luna`). Results are written to an Excel workbook with retrieval metrics and columns for manual scoring.

## Setup

1. **Install dependencies** 

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

2. **Configure credentials.** 
   Create a .env to store your enpoints

   | Variable | Value |
   | --- | --- |
   | `AZURE_OPENAI_ENDPOINT` |  e.g. `https://<resource>.openai.azure.com` (`/openai/v1/` is added automatically) |
   | `AZURE_OPENAI_API_KEY` | Key from the deployment page in Azure AI Foundry |
   | `AZURE_CHAT_DEPLOYMENT` | Deployment name, default `gpt-6-luna` |
   | `EMBEDDING_MODEL` | Default `BAAI/bge-small-en-v1.5` (downloads once, runs on CPU) |

3. **Add the inputs.**
   - Place the corpus PDFs in a folder, e.g. `./docs`.
   - Provide the question workbook (e.g. `omega_rag_eval_questions.xlsx`) with a `Questions` sheet containing `ID`, `Question`, `Expected answer`, `Source document`, and `Page` columns.


## Running

python rag_eval.py --docs docs --questions omega_rag_eval_questions.xlsx


| Option | Default | Description |
| --- | --- | --- |
| `--chunk-size` | 800 | Characters per chunk (keep at or below ~1200 to stay within the embedding model's 512-token limit) |
| `--overlap` | 15% of chunk size | Characters shared between neighboring chunks |
| `--k` | 4 | Chunks retrieved per question |
| `--no-prefix` | off | Removes the document-title header from each chunk |
| `--skip-baseline` | off | Skips the no-retrieval baseline |
| `--dry-run` | off | Tests the pipeline without model calls |
| `--out` | auto | Output file name |

## Outputs

Each run writes `results_cs<chunk>_<prefix|noprefix>_k<k>.xlsx` with four sheets:

- **Summary** – mean precision@k, recall@k, and answer-quality totals (updates as you score).
- **Results** – RAG and baseline answers per question, retrieval metrics, and yellow columns for manual scoring.
- **Retrieved chunks** – the exact text passed to the model, for checking faithfulness and diagnosing failures.
- **Run config** – model, embedding model, chunking settings, and corpus used.

Embeddings are cached in `chroma_db/` per configuration, so repeated runs do not re-embed the documents.

