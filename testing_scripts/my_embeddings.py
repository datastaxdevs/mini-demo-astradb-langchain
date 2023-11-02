import json
from typing import List

from langchain.embeddings.base import Embeddings


class StupidEmbeddings(Embeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        unnormed0 = [ord(c) for c in text[:100]]
        unnormed = (unnormed0 + [1] + [0] * (100 - 1 - len(unnormed0)))[:100]
        norm = sum(x*x for x in unnormed)**0.5
        normed = [x/norm for x in unnormed]
        return normed

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class ParserEmbeddings(Embeddings):

    def __init__(self, dimension):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            vals = json.loads(text)
            assert len(vals) == self.dimension
            return vals
        except Exception:
            print(f"[ParserEmbeddings] Returning a moot vector for \"{text}\"")
            return [0.0] * self.dimension

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)
