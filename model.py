import random as rnd
from scores import number_of_prediction, create_dict_binare, create_dict_scores, mean_csv_result
from copy import deepcopy
import time
import math


    
def evaluate_weights(raw_dict: dict, actual_survived: dict, weights: dict, treashold):
    binare_dict = predict_dataset(raw_dict, weights, treashold)[1]
    n = number_of_prediction(binare_dict, actual_survived)/len(binare_dict)
    return n

def best_threshold(raw_dict: dict, 
                   actual_survived: dict, 
                   steps: int, 
                   train_weights: dict, 
                   start_threashold) :
  
    best_threshold = start_threashold
    beast_evaluate = 0
    range_values = float_range(start_threashold - start_threashold/2, start_threashold + start_threashold/2, steps,3)
    for value in range_values:
        evaluate = evaluate_weights(raw_dict, actual_survived, train_weights, value)
        if evaluate > beast_evaluate:
            beast_evaluate = evaluate
            best_threshold = value
    return best_threshold


def predict_dataset(raw_dict, weights: dict, threshold: float)-> tuple:
    """Инциализация стартовых словарей"""
    
    score_dict = create_dict_scores(raw_dict, weights)
    binare_dict = create_dict_binare(score_dict, threshold)
    return score_dict, binare_dict


def try_improve_weights(raw_dict: dict, 
                        actual_survived: dict, 
                        weights: dict, step: 
                        float, 
                        threshold: float):
    """Возвращает словарь новых весов если accurcay стало лучше"""
    
    start_accuracy = evaluate_weights(raw_dict, actual_survived, weights, threshold)
    new_weights = create_new_weights(weights, step)
    new_accuracy = evaluate_weights(raw_dict, actual_survived, new_weights, threshold)
    if new_accuracy > start_accuracy:
        return new_weights, True
    return weights, False
    
def create_new_weights(weights: dict, step: float) -> dict:
    """Возвращает новый словарь с весами, одно значение которого увеличилось или уменьшилась на step"""
    weights_copy = deepcopy(weights)
    all_features = [
    (feature, value)
    for feature, values in weights.items()
    for value in values
    if value != "max_key"]
    feature1, value1 = rnd.choice(all_features)
    all_features.remove((feature1, value1))
    feature2, value2 = rnd.choice(all_features)
    weights_copy[feature1][value1] += step
    feature2, value2 = rnd.choice(all_features)
    weights_copy[feature2][value2] -= step
    return weights_copy

def float_range(start: float, end: float, steps: int, round_value: int) -> list[float]:
    """Возвращает спсиок дробных значений от start (включительно) до end не включительно"""
    return [round(start + i*((end - start)/steps), round_value) for i in range(steps)]

def train_classifier(raw_dict: dict, 
                     actual_survived: dict, 
                     weights: dict, 
                     steps: float, 
                     iters: int,
                     treashold: float):
    "Подбираем лучшие веса для классифкатора"
    new_weights = deepcopy(weights)

    start_time = time.perf_counter()
    for step in steps:
        i = 0
        while i <= iters:
            new_weights, new = try_improve_weights(raw_dict, actual_survived,  new_weights, step, treashold)
            if new:
                i = 0
            else:
                i += 1
    end_time = time.perf_counter() 
    t = round(end_time - start_time,2)
    return new_weights, t
    
   
        