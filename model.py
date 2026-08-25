import random as rnd
from scores import number_of_prediction, create_dict_binary, create_dict_scores, log_loss, sigmoid
from copy import deepcopy
import time


def float_range(start: float, end: float, steps: int, round_value: int) -> list[float]:
    """Возвращает спсиок дробных значений от start (включительно) до end не включительно"""
    return [round(start + i*((end - start)/steps), round_value) for i in range(steps)]

def calculate_mean_loss(raw_list: list,
                     survived: dict,
                     weights: dict):
    sum_loss = 0
    length = len(raw_list)
    scores = create_dict_scores(raw_list, weights)
    for id, score in scores.items():
        sum_loss += log_loss(survived[id], sigmoid(score))

    return sum_loss/length

def predict_dataset(raw_list: list, weights: dict, threshold: float)-> tuple:
    """Инциализация стартовых словарей"""
    
    score_dict = create_dict_scores(raw_list, weights)
    binare_dict = create_dict_binary(score_dict, threshold)
    return score_dict, binare_dict

def select_weights(raw_list: list, 
                   actual_survived: dict, 
                   weights: dict, 
                   step: float):
    """Возвращает словарь новых весов если mean_loss стало меньше"""
    
    mean_loss = calculate_mean_loss(raw_list, actual_survived, weights)
    new_weights = create_new_weights(weights, step)
    new_mean_loss = calculate_mean_loss(raw_list, actual_survived, new_weights)
    if new_mean_loss < mean_loss:
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

def train_classifier(raw_list: list, 
                     actual: dict, 
                     weights: dict, 
                     steps: float, 
                     iters: int) -> tuple[dict, float]:
    "Подбираем лучшие веса для классифкатора"
    new_weights = deepcopy(weights)

    start_time = time.perf_counter()
    for step in steps:
        i = 0
        while i <= iters:
            new_weights, new = select_weights(raw_list, actual,  new_weights, step)
            if new:
                i = 0
            else:
                i += 1
    end_time = time.perf_counter() 
    t = round(end_time - start_time,2)
    return new_weights, t
    
   
        