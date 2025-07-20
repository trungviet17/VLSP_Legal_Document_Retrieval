import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 


from core.retriever.baseline_retriever import BaseRetriever
from core.db.qdrant import QdrantConnector
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
import numpy as np 
from typing import List, Dict


class FusionRetriever(BaseRetriever):

    def __init__(self, db_connector: QdrantConnector, corpus_path: str, limit: int,  alpha: float = 0.7, vector_threshold: float = 0.5):
        super().__init__(db_connector)
        self.corpus = self._load_corpus(corpus_path)
        self.alpha = alpha
        self.limit = limit
        self.vector_threshold = vector_threshold


    def _load_corpus(self, corpus_path: str) : 
        
        self.corpus_texts = []
        self.corpus_metadata = []

        with open(corpus_path, 'r', encoding='utf-8') as file:
            corpus = file.readlines()


        tokenized_corpus = []
        for i in corpus: 

            text =ViTokenizer.tokenize(i.get("law_context", "")) 
            tokens = text.split()
            tokenized_corpus.append(tokens)

            self.corpus_texts.append(text)
            self.corpus_metadata.append({
                "law_id": i.get("law_id", ""),
                "child_id": [sub_i.get("aid", None) for sub_i in i.get("context", [])],
            })


        self.bm25 = BM25Okapi(tokenized_corpus)


    def _bm25_retrieve(self, query: str): 

        tokenized_query = ViTokenizer.tokenize(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.limit * 2]

        results = []

        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "law_id": self.corpus_metadata[idx].get("law_id", ""),
                    "score": scores[idx],
                    "text": self.corpus_texts[idx],
                    "child_id": self.corpus_metadata[idx].get("child_id", []), 
                    "bm25_score": float(scores[idx]) 
                })

        return results 
    


    def _vector_retrieve(self, query: str) -> list: 

        tokenized_query = ViTokenizer.tokenize(query)
        
        response = self.qdrant.query(
            query=tokenized_query, 
            limit=self.limit * 2, 
            threshold=self.vector_threshold
        )


        results = []
        for item in response.get("result", []):
            results.append({
                "law_id": item.get("law_id", ""),
                "text": item.get("text", ""),
                "child_id": item.get("child_id", []),
                "vector_score": float(item.get("score", 0.0))

            })
        
        return results 
    


    def combine_retrieve(self, bm25_results: List[Dict], vector_results: List[Dict]) -> list:
        
        # normalize score 
        if bm25_results: 
            max_bm25_score = max([item['bm25_score'] for item in bm25_results])
            min_bm25_score = min([item['bm25_score'] for item in bm25_results])

            if max_bm25_score > min_bm25_score: 
                for item in bm25_results:
                    item['bm25_score'] = (item['bm25_score'] - min_bm25_score) / (max_bm25_score - min_bm25_score)
            else:
                for item in bm25_results:
                    item['bm25_score'] = 1.0


        if vector_results:
            max_vector_score = max([item['vector_score'] for item in vector_results])
            min_vector_score = min([item['vector_score'] for item in vector_results])

            if max_vector_score > min_vector_score:
                for item in vector_results:
                    item['vector_score'] = (item['vector_score'] - min_vector_score) / (max_vector_score - min_vector_score)
            else:
                for item in vector_results:
                    item['vector_score'] = 1.0

        # combine results
        combined = {}
        for result in bm25_results:
            doc_id = result['id']
            combined[doc_id] = {
                'id': doc_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'bm25_score': result.get('bm25_score', 0.0),
                'vector_score': 0.0
            }

        for result in vector_results:

            doc_id = result['id']
            if doc_id in combined: 
                combined[doc_id]['vector_score'] = result.get('vector_score', 0.0)
            else:
                combined[doc_id] = {
                    'id': doc_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'bm25_score': 0.0,
                    'vector_score': result.get('vector_score', 0.0)
                }

        for doc_id in combined: 
            combined[doc_id]['score'] = (
                self.alpha * combined[doc_id]['bm25_score'] + 
                (1 - self.alpha) * combined[doc_id]['vector_score']
            )

        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)

        return sorted_results[:self.limit] if self.limit > 0 else sorted_results
    

    def retrieve(self, query: str) -> list:

        bm25_results = self._bm25_retrieve(query)
        vector_results = self._vector_retrieve(query)

        combined_results = self.combine_retrieve(bm25_results, vector_results)

        return combined_results 






    