from typing import List 
import numpy as np 


class MetricCalculator:
    

    @staticmethod 
    def recall(ground_truth: List[int], pred: List[int]) -> float:
        
        if not ground_truth:return 0.0
        
        gt_set = set(ground_truth)
        pred_set = set(pred)

        true_positives = len(gt_set.intersection(pred_set))
        false_negatives = len(gt_set - pred_set)

        if true_positives + false_negatives == 0: return 0.0 

        return true_positives / (true_positives + false_negatives)


    @staticmethod
    def precision(ground_truth: List[int], pred: List[int]) -> float:
        if not pred : return 0.0         
        
        gt_set = set(ground_truth)
        pred_set = set(pred)

        true_positives = len(gt_set.intersection(pred_set))
        false_positives = len(pred_set - gt_set)

        if true_positives + false_positives == 0: return 0.0

        return true_positives / (true_positives + false_positives)

        
    @staticmethod
    def f2_score(ground_truth: List[int], pred: List[int]) -> float:
        recall = MetricCalculator.recall(ground_truth, pred)
        precision = MetricCalculator.precision(ground_truth, pred)

        if recall + precision == 0: return 0.0

        beta = 2
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    

    @staticmethod
    def macro_f2(ground_truth_list: List[List[int]], pred_list: List[List[int]]) -> float:
        if len(ground_truth_list) != len(pred_list):
            raise ValueError("Ground truth and prediction lists must have the same length.")

        f2_scores = []
        for ground_truth, pred in zip(ground_truth_list, pred_list):
            f2_scores.append(MetricCalculator.f2_score(ground_truth, pred))

        return np.mean(f2_scores) if f2_scores else 0.0