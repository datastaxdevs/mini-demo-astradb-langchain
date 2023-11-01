import os

from langchain.vectorstores import AstraDB
from langchain.schema import Document

from my_embeddings import StupidEmbeddings
from my_timer import MyTimer

COLLECTION_NAME = "astra_lc_minitests_drop"


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

    # we invoke DROP
    vstore_kamikaze = AstraDB(
        embedding=embe,
        collection_name=COLLECTION_NAME,
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    vstore_kamikaze.delete_collection()

    # we try to read from the table, lol
    try:
        results = vstore.similarity_search_with_score(str(N+1/7.0), k=2)
        for res, score in results:
            print(f"* [SIM={score:3f}] '{res.page_content}'")
    except ValueError as e:
        if f"table {COLLECTION_NAME} does not exist" in str(e):
            print("Expected error: drop works")
        else:
            print("Unexpected error: INVESTIGATE")
            print(str(e))
