import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.db.qdrant import QdrantConnector
from pyvi import ViTokenizer
from src.core.retriever.baseline_retriever import BaseRetriever
from sentence_transformers import CrossEncoder

class RerankRetriever(BaseRetriever): 

    def __init__(self, db_connector: QdrantConnector, cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", rerank_top_k: int = 3): 
        super().__init__(db_connector)
        self.cross_encoder_name = cross_encoder_name
        self.cross_encoder = CrossEncoder(self.cross_encoder_name)
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: list[str], limit: int = 5, threshold: float = 0.5) -> dict:
      
        query = ViTokenizer.tokenize(query)
        response = self.qdrant.query(
            query=query, 
            limit=limit, 
            threshold=threshold
        )

        # Prepare pairs for cross-encoder
        pairs = [[query, doc['payload']['text']] for doc in response]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort documents by score
        scored_docs = sorted(zip(response, scores), key=lambda x: x[1], reverse=True)

        # Top reranked documents
        docs = [doc for doc, _ in scored_docs[:self.rerank_top_k]]

        return docs