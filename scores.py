"""Модуль отвечает за математику предсказания"""

from config import (FEATURES)
import math


def log_loss(s: int, p: float) -> float:
    """Ошибка для одного пассажира"""
    p = max(min(p,0.999),0.001)
    
    return -(s*math.log(p) + (1-s)*math.log(1-p))


def sigmoid(predict: float):
    return 1/(1+pow(math.e,-predict))

def summ_weights(wights: dict) -> float:
    """Считает сумму весов"""
    res = 0
    for category, features in wights.items():
        for value in features.values():
            res += value
    return res


def scores_to_sigmoids(scores: dict) -> dict:
    return dict((id,sigmoid(score)) for id, score in scores.items())
def mean_csv_result(scores_dict: dict) -> float:
    """Считает среднее значение score"""
    result = 0.0
    for k, v in scores_dict.items():
        result += float(v)
    return result / len(scores_dict)
        
def binarize_score(score, threshold) -> int:
    return int(score >= threshold)

def get_score(passenger: dict, weights: dict) -> float:
    """Считаем итоговое значение для одного пассажира"""
    score = 0.0
    for feature in FEATURES:
        if feature == "Bias":
            score += weights["Bias"]["bias"]
            continue
        
        values = passenger[feature]
        if values == "":
          continue
        
        if values not in weights[feature]:
                score += 0
        else:
            score += float(weights[feature][values])      
    return score


def create_dict_scores(dataset: dict, weights: dict) -> dict:
    """Собирает словарь с предиктом для каждого пасажира"""
    result = {}
    for passenger in dataset:
        result[passenger["PassengerId"]] = get_score(passenger,weights)
        
    return result


def create_dict_binary(dataset: dict, threshold: float) -> dict:
    """Собирает словарь бинарными значениями для каждого пасажира"""
    result = {}
    for id, score in dataset.items():
        result[id] = binarize_score(float(score), threshold)
    return result
    

def survived_counter(file_dict: dict) -> int:
    """Считает количекство выживших в датесете file_name"""
    counter = 0
    for k, value in file_dict.items():
        counter += int(value)
    return counter

def number_of_prediction(binare_dict: dict, survived_dict: dict) -> int:
    """Считаем количество угадываний"""
    counter = 0
    for id, _is_survived in survived_dict.items():
        counter += int(binare_dict[id] == _is_survived)
    return counter

def percent_prediction(lenth, number):
    return round(100*(number/lenth),2)

#BASELINE

def number_of_baseline(baseline_binare: dict, actual_survived: dict) -> int:
    return number_of_prediction(baseline_binare, actual_survived)

# MODEL

