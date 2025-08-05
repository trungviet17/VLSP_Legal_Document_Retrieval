import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.db.qdrant import QdrantConnector
from pyvi import ViTokenizer
class BaseRetriever: 

    def __init__(self, db_connector: QdrantConnector, limit: int = 5, threshold: float = 0.5): 
        self.qdrant = db_connector
        self.limit = limit 
        self.threshold = threshold


    def retrieve(self, query: list[str]) -> dict:
      
        query = ViTokenizer.tokenize(query)
        response = self.qdrant.query(
            query=query, 
            limit=self.limit, 
            threshold=self.threshold
        )
        return response
    