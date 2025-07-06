import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db.qdrant import QdrantConnector 
from core.retriever.baseline_retriever import BaseRetriever
from core.chunking.baseline_chunking import BaseChunker
from omegaconf import DictConfig
from benchmark.cal_metric import MetricCalculator 
import json 
from tqdm import tqdm
from datetime import datetime
import hydra 

class BasePipeline:

    def __init__(self, cfg: DictConfig): 
        self.cfg = cfg

        self.chunker = BaseChunker(
            max_tokens = cfg.chunking.max_tokens
        )   

        self.db = QdrantConnector(
            collection_name = cfg.db.collection_name,
            vector_size = cfg.embedding.vector_size, 
            embedding_model_name = cfg.embedding.embedding_model, 
            embedding_type = cfg.embedding.embedding_type, 
            embedding_cache_dir = cfg.cache_dir
        )

        self.retriever = BaseRetriever(
            db_connector = self.db
        )

        os.makedirs(os.path.dirname(cfg.output.result_path), exist_ok=True)

    
    def _load_data_(self, file_path: str): 
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data


    def process_corpus(self): 
        self.corpus = self._load_data_(self.cfg.data.corpus_path)

        texts = []
        metadata = []

        for doc in tqdm(self.corpus, desc="Processing corpus"):
            
            for sub_doc in doc['content']: 

                texts.append(sub_doc['content_Article']) 
                metadata.append({
                    'id': doc['id'], 
                    'aid': sub_doc['aid'],
                    'law_id': doc['law_id'],
                    
                })
                
        self.chunks = self.chunker.process_corpus(texts)

        self.db.embedd_chunks(
            chunks = self.chunks, 
            metadata = metadata
        )

    def query_data(self): 

        self.eval_data = self._load_data_(self.cfg.data.train_path)

        results = []
        

        for item in tqdm(self.eval_data, desc="Querying data"):
            query = item['question']

            ground_truth = item['relevant_laws']

            retrieved_docs = self.retriever.retrieve(
                query = query, 
                threshold = self.cfg.retrieval.threshold, 
                limit = self.cfg.retrieval.limit
            )

            retrieved_ids = [doc['aid'] for doc in retrieved_docs]

            results.append(
                {
                    'query': query,
                    'ground_truth': ground_truth,
                    'retrieved': retrieved_ids
                }
            )

        return results  


    def evaluate_and_save_results(self, results):

        recall = [MetricCalculator.recall(result['ground_truth'], result['retrieved']) for result in results]
        precision = [MetricCalculator.precision(result['ground_truth'], result['retrieved']) for result in results]
        f2_scores = [MetricCalculator.f2_score(result['ground_truth'], result['retrieved']) for result in results]

        macro_recall = sum(recall) / len(recall) if recall else 0
        macro_precision = sum(precision) / len(precision) if precision else 0
        macro_f2 = MetricCalculator.macro_f2(recall, precision) if recall and precision else 0

        print("Saving results to", self.cfg.output.result_path)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        output_data = {
            "timestamp": timestamp,
            "configuration": {
            "collection": self.cfg.db.collection_name,
            "embedding_model": self.cfg.embedding.embedding_model,
            "chunk_size": self.cfg.chunking.max_tokens,
            "retrieval_limit": self.cfg.retrieval.limit,
            "threshold": self.cfg.retrieval.threshold
            },
            "metrics": {
            "macro_recall": macro_recall,
            "macro_precision": macro_precision,
            "macro_f2": macro_f2
            },
            "per_query_results": [
            {
                "query": res['query'],
                "ground_truth": res['ground_truth'],
                "retrieved": res['retrieved'],
                "recall": rec,
                "precision": prec,
                "f2": f2
            } for res, rec, prec, f2 in zip(results, recall, precision, f2_scores)]
        }

        with open(self.cfg.output.result_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return {
            "macro_recall": macro_recall,
            "macro_precision": macro_precision,
            "macro_f2": macro_f2
        }


    def run(self): 
        self.process_corpus()
        results = self.query_data()
        evaluation_results = self.evaluate_and_save_results(results)
        return evaluation_results




@hydra.main(config_path = "../config", config_name = "default")
def main(cfg: DictConfig):
    pipeline = BasePipeline(cfg)
    results = pipeline.run()
    print("Evaluation Results:", results)


if __name__ == "__main__":
    main()