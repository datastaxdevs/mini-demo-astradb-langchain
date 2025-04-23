# mini-demo-astradb-langchain

For more information, visit the DataStax [Astra DB docs page](https://docs.datastax.com/en/astra-db-serverless/integrations/langchain.html).

[Open in Colab](https://colab.research.google.com/github/datastaxdevs/mini-demo-astradb-langchain/blob/main/AstraDB_langchain_quickstart_1.ipynb)

## Alternatively, run locally 

Install `Jupyter`.

Export the following environment variables if desired:

```
ASTRA_DB_API_ENDPOINT="https://..."
ASTRA_DB_APPLICATION_TOKEN="AstraCS:..."

ASTRA_DB_KEYSPACE="..."             # OPTIONAL

OPENAI_API_KEY="..."                # OPTIONAL (required if using explicit embeddings)
ASTRA_DB_API_KEY_NAME="..."         # OPTIONAL (required if using 'vectorize')
```

Open in Jupyter and run each cell.

_(Requires Python version 3.9 or higher.)_
