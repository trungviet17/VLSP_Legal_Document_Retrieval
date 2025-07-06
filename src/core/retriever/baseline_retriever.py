import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.db.qdrant import QdrantConnector
from pyvi import ViTokenizer
class BaseRetriever: 

    def __init__(self, db_connector: QdrantConnector): 
        self.qdrant = db_connector


    def retrieve(self, queries: list[str], limit: int = 5, threshold: float = 0.5) -> list[dict]:
      
        results = []
        for query in queries:
            query = ViTokenizer.tokenize(query)
            response = self.qdrant.query(
                query=query, 
                limit=limit, 
                threshold=threshold
            )
            results.extend(response)
        return results