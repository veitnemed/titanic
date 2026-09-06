from storage import (survived_dict,
                     baseline_dict, 
                     get_train_csv_lists,
                     save_json,
                     load_json, 
                     json_init)
from config import  (DATASET_CSV_PATCH, 
                     DEFAULT_WEIGHTS,
                     WEIGHT_STEPS,
                     NO_CHANGE_ITERATIONS,
                     WEIGHTS_JSON_PATCH,
                     SPLIT_SEED,
                     TRAIN_SEED)
from features import replace_median_ages
from info import print_result
from model import train_classifier, calculate_mean_loss
import os

def init_train_dicts(train_data: list, test_data: list) -> dict:
    
    default_weights = DEFAULT_WEIGHTS.copy()
    age_values = None
    if "Age" in default_weights:
        age_values = list(map(int,list(default_weights["Age"].keys())))
    # 1) Начальные рассчёты
    json_init(WEIGHTS_JSON_PATCH, DEFAULT_WEIGHTS)
    
    features_list = list(default_weights.keys())
    start_weights = load_json(WEIGHTS_JSON_PATCH)
    train_answers = survived_dict(train_data)
    baseline_predictions = baseline_dict(train_data)
    valid_answers = survived_dict(test_data)
    start_loss = calculate_mean_loss(train_data, train_answers, start_weights, features_list, age_values)
                   
    return  {"train_data": train_data,
             "test_data": test_data,
             "start_weights": start_weights, 
             "train_answers": train_answers,
             "start_loss": start_loss,
             "valid_answers": valid_answers,
             "features_list": features_list,
             "age_values": age_values,
             "baseline_predictions": baseline_predictions}


def impact_weights(test, train, start_loss, seed):
    
    print("ВКЛАДЫ ВЕСОВ\n")
    print(f"Изначальный LOSS: {round(start_loss, 3)}")
    print(f"Seed: {seed}")
    
    default_weights = DEFAULT_WEIGHTS.copy()
    weights = DEFAULT_WEIGHTS
    res = {}
    
    for class_feature, feature in DEFAULT_WEIGHTS.items():
    
        default_weights.pop(class_feature)
        
        if "Age" in default_weights:
            age_values = list(map(int,list(default_weights["Age"].keys())))
            
        weights, survived, start_loss, survived_test, features_list, age_values, baseline = init_train_dicts(train,test)
        train_weights,_ = train_classifier(raw_list = train,
                                          train_answers = survived,
                                          start_weights = weights,
                                          steps = WEIGHT_STEPS,
                                          iters = NO_CHANGE_ITERATIONS,
                                          seed_value = seed,
                                          features_list = features_list,
                                          age_values = age_values)  
        
        new_loss = calculate_mean_loss(test, survived_test, train_weights, features_list, age_values)
        res[class_feature] = round(new_loss,3)
        default_weights[class_feature] = feature
    
    for class_feature, feature in res.items():
        print(f"Loss без {class_feature}: {feature} // Польза {100*round(feature-start_loss,4)} % ")
        
def mean_impact_weights(test: list, train: list):
    for idx, seed in enumerate(list(range(10))):
            print(f"Эксперимент {idx+1}")
            start_loss = train_model(test,train,seed)
            impact_weights(start_loss,seed)
            
def train_model(valid_data: list, 
             train_data: list, 
             seed: int, 
             is_show_progress: bool = False, 
             is_show_result: bool = False) -> dict:
    """Обучение, обновыление весов и вывод losss"""
    
    train_context = init_train_dicts(train_data = train_data, test_data = valid_data)
    weights_train, time_train = train_classifier(raw_list = train_data,
                                              train_answers = train_context["train_answers"],
                                              start_weights = train_context["start_weights"],
                                              steps = WEIGHT_STEPS,
                                              iters = NO_CHANGE_ITERATIONS,
                                              seed_value = seed,
                                              features_list = train_context["features_list"],
                                              age_values = train_context["age_values"],
                                              is_show_progress = is_show_progress)
    
    loss_train = calculate_mean_loss(valid_data,train_context["valid_answers"],weights_train,train_context["features_list"], train_context["age_values"])
    if is_show_result is True:
        print_result(train_data = train_data,
                    valid_data = valid_data,
                    seed = seed,
                    start_weights = train_context["start_weights"],
                    trained_weights = weights_train,
                    baseline_predictions = train_context["baseline_predictions"],
                    train_answers = train_context["train_answers"],
                    valid_answers = train_context["valid_answers"],
                    features_list = train_context["features_list"],
                    age_values = train_context["age_values"],
                    start_loss = train_context["start_loss"],
                    train_loss = loss_train,
                    time_train = time_train
                    )
    return {
        "weights_train": weights_train,
        "loss_train": loss_train,
        "time_train": time_train
    }
    
def main_func():
    train, test  = get_train_csv_lists(DATASET_CSV_PATCH, seed_value = SPLIT_SEED)
    train, test = replace_median_ages(train), replace_median_ages(test)
    res_train = train_model(valid_data = test, 
                         train_data = train, 
                         seed = TRAIN_SEED,
                         is_show_result = True, 
                         is_show_progress=True)
    save_json(WEIGHTS_JSON_PATCH, res_train["weights_train"])
    
if __name__ == "__main__":
    os.system("cls")
    main_func()
    

