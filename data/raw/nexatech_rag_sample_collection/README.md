# NexaTech Sample Document Collection

This is a fictional dataset created for learning and testing a document-based RAG pipeline.

## Contents
- 15 sample files
- PDF, DOCX, TXT, and CSV formats
- Fictional HR, IT, Engineering, Support, Finance, Operations, Sales, Legal, and Project Management content

## Important
All company names, policies, service limits, performance figures, tickets, and records are fictional.
Do not treat these as real policies or operational guidance.

## Suggested first ingestion test
Start with `data/raw/` and implement a recursive file scanner. Then extract document text and metadata, chunk the extracted text, generate embeddings, and store the chunks in ChromaDB.
