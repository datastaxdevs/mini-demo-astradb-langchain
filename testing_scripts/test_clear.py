import os

from langchain.vectorstores import AstraDB
from langchain.schema import Document

from my_embeddings import StupidEmbeddings
from my_timer import MyTimer

COLLECTION_NAME = "astra_lc_minitests_clear"


if __name__ == '__main__':
    embe = StupidEmbeddings()
    vstore = AstraDB(
        embedding=embe,
        collection_name=COLLECTION_NAME,
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    # insert N embeddings
    N = 10
    texts = [
        str(i+1/7.0)
        for i in range(N)
    ]
    ids = ["doc_%i" % i for i in range(N)]
    with MyTimer(f"adding {len(texts)}"):
        vstore.add_texts(texts=texts, ids=ids)

    results = vstore.similarity_search_with_score(str(N+1/7.0), k=2)
    assert len(results) == min(2, N)

    # we invoke clear
    vstore.clear()

    results = vstore.similarity_search_with_score(str(N+1/7.0), k=2)
    assert results == []
