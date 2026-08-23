import random as rnd
from scores import number_of_prediction, create_dict_binare, create_dict_scores, mean_csv_result
from copy import deepcopy

def evaluate_weights(raw_dict: dict, actual_survived: dict, weights: dict):
    binare_dict = predict_dataset(raw_dict, weights)[1]
    n = number_of_prediction(binare_dict, actual_survived)/len(binare_dict)
    return n


def predict_dataset(raw_dict, weights: dict)-> tuple:
    """Инциализация стартовых словарей"""
    
    score_dict = create_dict_scores(raw_dict, weights)
    threshold = mean_csv_result(score_dict)
    binare_dict = create_dict_binare(score_dict, threshold)
    return (score_dict, binare_dict, threshold)

def try_improve_weights(raw_dict: dict, actual_survived: dict, weights: dict, step: float):
    """Возвращает словарь новых весов если accurcay стало лучше"""
    start_accuracy = evaluate_weights(raw_dict, actual_survived, weights)
    new_weights = create_new_weights(weights, step)
    new_accuracy = evaluate_weights(raw_dict, actual_survived, new_weights)
    if new_accuracy > start_accuracy:
        return new_weights
    return weights
    
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


        
def train_classifier(raw_dict: dict, actual_survived: dict, weights: dict, step: float, iters: int):
    "Подбираем лучшие веса для классифкатора"
    new_weights = deepcopy(weights)
    for i in range(iters):
        if i % 10 == 0:
            print(f"{i}/{iters}")
        new_weights = try_improve_weights(raw_dict, actual_survived,  new_weights, step)
    return new_weights
    
   
        