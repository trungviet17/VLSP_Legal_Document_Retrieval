import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.envconfig import EnvConfig
from qdrant_client import QdrantClient
from utils.model import get_embedding_model
from typing import Any, List, Dict
from qdrant_client.models import PointStruct, Distance, VectorParams 
from uuid import uuid4
from tqdm import tqdm


class QdrantConnector: 

    def __init__(self, collection_name: str, vector_size: int,  embedding_model_name: str = "all-MiniLM-L6-v2", embedding_type : str = "transformer", embedding_cache_dir: str = EnvConfig.CACHE_DIR,
                    qdrant_url : str = EnvConfig.QDRANT_URL,
                    qdrant_api_key : str = EnvConfig.QDRANT_API_KEY,  
                    ):  
        
        self.client = QdrantClient(
            url = qdrant_url,
            api_key = qdrant_api_key,
        ) 

        self.collection_name = collection_name
        self.embedding_model = get_embedding_model(
            model_name=embedding_model_name, 
            type=embedding_type, 
            cache_dir=embedding_cache_dir
        )
        self.vector_size = vector_size
        self._init_collection_()


    def _init_collection_(self): 
        collections = self.client.get_collections().collections 
        if self.collection_name not in [col.name for col in collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size = self.vector_size,
                    distance=Distance.COSINE,  
                )
            )
        else:
            print(f"Collection {self.collection_name} already exists.")



    def embedd_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 1000):

        points = []
        for i, (chunk) in tqdm(enumerate(zip(chunks)), total=len(chunks), desc="Embedding and preparing points"):
            meta = chunk.get('metadata', None)

            meta['text'] = chunk['text']
            print("hehe")
            vector = chunk['vector']
            point = {
                "id": str(uuid4()),
                "vector": vector,
                "payload": meta, 
            
            }
            points.append(PointStruct(**point))

            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                points = []
                
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        


    def query(self, query: str, threshold: float = 0.7, limit: int = 10):

        query_vector = self.embedding_model.embed_query(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=threshold
        )

        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            results.append(result)

        return results


    
    
    

    

