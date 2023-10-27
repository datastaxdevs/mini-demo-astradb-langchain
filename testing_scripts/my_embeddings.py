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