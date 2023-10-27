import os

from langchain.vectorstores import AstraDB
from langchain.schema import Document

from my_embeddings import StupidEmbeddings
from my_timer import MyTimer


if __name__ == '__main__':
    embe = StupidEmbeddings()
    vstore = AstraDB(
        embedding=embe,
        collection_name="astra_lc_minitests",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    # insert N embeddings
    N = 100
    texts = [
        str(i+1/7.0)
        for i in range(N)
    ]
    ids = ["doc_%i" % i for i in range(N)]
    with MyTimer(f"adding {len(texts)}"):
        vstore.add_texts(texts=texts, ids=ids)

    # deleting some/all of these
    ids_to_delete = ids + # ['nonexisting']
    print(f"deleting {len(ids_to_delete)} ...")
    with MyTimer("Deletion"):
        result = vstore.delete(ids_to_delete)
        print(f"deletion => {result}")

'''
need this (truncate)?
    def clear(self) -> None:

need this? (found on chromadb)
    def delete_collection(self) -> None:

use this to delete based on some md attributes?
    {
      "deleteMany": {
        "filter": {
          "status": "inactive"
        }
      }
    }
'''
