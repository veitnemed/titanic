from storage import (survived_dict,
                     baseline_dict, 
                     get_train_csv_lists,
                     save_json,
                     load_json, 
                     json_init)
from config import  (DATASET_CSV_PATH, 
                     DEFAULT_WEIGHTS,
                     WEIGHT_STEPS,
                     NO_CHANGE_ITERATIONS,
                     WEIGHTS_JSON_PATH,
                     SPLIT_SEED,
                     TRAIN_SEED,
                     SEEDS_RANGE)
from features import replace_median_ages
from info import print_result, show_top_n_error
from model import train_classifier, calculate_mean_loss
import os

def init_train_dicts(train_data: list, valid_data: list, default_weights = DEFAULT_WEIGHTS) -> dict:
    
    age_values = None
    if "Age" in default_weights:
        age_values = list(map(int,list(default_weights["Age"].keys())))
    # 1) Начальные рассчёты
    json_init(WEIGHTS_JSON_PATH, DEFAULT_WEIGHTS)
    
    features_list = list(default_weights.keys())
    start_weights = load_json(WEIGHTS_JSON_PATH)
    train_answers = survived_dict(train_data)
    baseline_predictions = baseline_dict(train_data)
    valid_answers = survived_dict(valid_data)
    start_loss = calculate_mean_loss(train_data, train_answers, start_weights, features_list, age_values)
                   
    return  {"train_data": train_data,
             "valid_data": valid_data,
             "start_weights": start_weights, 
             "train_answers": train_answers,
             "start_loss": start_loss,
             "valid_answers": valid_answers,
             "features_list": features_list,
             "age_values": age_values,
             "baseline_predictions": baseline_predictions}

def print_ablation_result(full_model_valid_loss: float, seed_amount: int, mean_losses_without_feature: dict):
    """Вывод абляционного теста """
    print("ВКЛАДЫ ВЕСОВ\n")
    print(f"Колчиество seed по которым, усреднялся loss: {seed_amount}")
    
    print(f"Изначальны LOSS: {round(full_model_valid_loss, 3)}")
    for tup in mean_losses_without_feature.items():
            
            class_feature, feature = tup
            feat_round = round(feature,3)
            delta_round = round(feature - full_model_valid_loss, 3)
            precent = round(100*(feature/full_model_valid_loss))
            print(f"Loss без {class_feature}: {feat_round}; Вклад: {delta_round} / {precent} ")
                
def ablation_weights_test(filename, seed):
    
    from copy import deepcopy
    train_data, valid_data  = get_train_csv_lists(filename, seed_value = seed)
    train_data, valid_data = replace_median_ages(train_data), replace_median_ages(valid_data)
    ablation_weights = deepcopy(DEFAULT_WEIGHTS)
    losses_result = {}
    for class_feature, feature in DEFAULT_WEIGHTS.items():
    
        ablation_weights.pop(class_feature)
        train_context = init_train_dicts(train_data, valid_data, ablation_weights)
        weights_train, _ = train_classifier(raw_list = train_data,
                                                      train_answers = train_context["train_answers"],
                                                      start_weights = ablation_weights,
                                                      steps = WEIGHT_STEPS,
                                                      iters = NO_CHANGE_ITERATIONS,
                                                      seed_value = seed,
                                                      features_list = train_context["features_list"],
                                                      age_values = train_context["age_values"],) 
        
        valid_loss = calculate_mean_loss(valid_data, train_context["valid_answers"],weights_train,train_context["features_list"], train_context["age_values"])
        losses_result[class_feature] = round(valid_loss,3)
        ablation_weights[class_feature] = feature
            
    return losses_result
        
def series_ablation(filename: str, seed_range = SEEDS_RANGE):
    from collections import defaultdict
    
    mean_losses_without_feature = defaultdict(int)
    tottal_lossed = []
    N = len(SEEDS_RANGE)
    for seed in seed_range:
        print(f"Рассчёт для SEED = {seed}")
        result_context = train_model(filename,seed)
        tottal_lossed.append(result_context["valid_loss"])
        losses_without_feature = ablation_weights_test(filename, seed)
        for class_feature, loss in losses_without_feature.items():
            mean_losses_without_feature[class_feature] += loss/N
    full_model_valid_loss = sum(tottal_lossed)/N
    print_ablation_result(full_model_valid_loss, 
                          seed_amount = N, 
                          mean_losses_without_feature = mean_losses_without_feature )
                
           
def train_model(filename: str,
             seed: int, 
             is_show_progress: bool = False, 
             is_show_result: bool = False,
             is_show_top_error: bool = False) -> dict:
    """Обучение, обновление весов и вывод loss"""
    
    train_data, valid_data  = get_train_csv_lists(filename, seed_value = seed)
    train_data, valid_data = replace_median_ages(train_data), replace_median_ages(valid_data)
    train_context = init_train_dicts(train_data = train_data, valid_data = valid_data)
    trained_weights, time_train = train_classifier(raw_list = train_data,
                                              train_answers = train_context["train_answers"],
                                              start_weights = DEFAULT_WEIGHTS,
                                              steps = WEIGHT_STEPS,
                                              iters = NO_CHANGE_ITERATIONS,
                                              seed_value = seed,
                                              features_list = train_context["features_list"],
                                              age_values = train_context["age_values"],
                                              is_show_progress = is_show_progress)

    valid_loss = calculate_mean_loss(valid_data,train_context["valid_answers"],trained_weights,train_context["features_list"], train_context["age_values"])
    result_context = {
        "trained_weights": trained_weights,
        "valid_loss": valid_loss,
        "time_train": time_train
    }
    if is_show_result is True:
        print_result(train_context = train_context,
                    result_context = result_context,
                    seed = seed,
                    )
    if is_show_top_error is True:
        show_top_n_error(train_data,
                         train_context["train_answers"],
                         trained_weights,
                         len(train_data) - 1,
                         train_context["features_list"],
                         train_context["age_values"])
    return result_context
    
def main_func():
    
    result_context = train_model(
                         DATASET_CSV_PATH,
                         seed = TRAIN_SEED,
                         is_show_result = True, 
                         is_show_progress = True,
                         is_show_top_error = False,)
    
    save_json(WEIGHTS_JSON_PATH, result_context["trained_weights"])
    #result_ablation = ablation_weights_test(valid_data, train_data, TRAIN_SEED)
    #print_ablation_result(tottal_loss = result_context["valid_loss"], 
                          #seed = SPLIT_SEED, 
                          #losses_result = result_ablation)
    #series_ablation(DATASET_CSV_PATH)
    
if __name__ == "__main__":
    os.system("cls")
    main_func()
    

