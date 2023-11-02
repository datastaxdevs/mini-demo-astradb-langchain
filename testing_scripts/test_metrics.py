import json
import os

from langchain.vectorstores import AstraDB
from langchain.schema import Document

from my_embeddings import ParserEmbeddings
from my_timer import MyTimer


if __name__ == '__main__':
    embe = ParserEmbeddings(dimension=2)

    dim = 2
    isq2 = 0.5**0.5
    isa = 0.7
    isb = (1.0 - isa*isa)**0.5
    texts = [
        json.dumps([isa, isb]),
        json.dumps([10*isq2, 10*isq2]),
    ]
    ids = [
        "[s-e, s+e]",
        "[10s, 10s]",
    ]

    query_text = json.dumps([isq2, isq2])

    vstore_cos = AstraDB(
        embedding=embe,
        collection_name="astra_lc_cos_test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        metric="cosine",
    )
    vstore_cos.add_texts(
        texts=texts,
        ids=ids,
    )
    result_cos = vstore_cos.similarity_search(query_text, k=1)[0]
    print("Cos => ", result_cos)

    vstore_euc = AstraDB(
        embedding=embe,
        collection_name="astra_lc_euc_test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        metric="euclidean",
    )
    vstore_euc.add_texts(
        texts=texts,
        ids=ids,
    )
    result_euc = vstore_euc.similarity_search(query_text, k=1)[0]
    print("Euc => ", result_euc)
