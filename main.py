from storage import (survived_dict,
                     baseline_dict, 
                     get_train_csv_lists,
                     save_json,
                     load_json, 
                     json_init)
from config import  (TRAIN, 
                     DEFAULT_WEIGHTS,
                     STEPS_FOR_TRAIN,
                     NUMBER_OF_ITERATIONS,
                     WEIGHTS,
                     SEED_SPLIT,
                     SEED_TRAIN)
from features import replace_median_ages
from info import show_result
from model import train_classifier, calculate_mean_loss
import os

def init_train_dicts(train: list, test: list) -> dict:
    
    default_weights = DEFAULT_WEIGHTS.copy()
    age_values = None
    if "Age" in default_weights:
        age_values = list(map(int,list(default_weights["Age"].keys())))
    # 1) Начальные рассчёты
    json_init(WEIGHTS, DEFAULT_WEIGHTS)
    
    features_list = list(default_weights.keys())
    weights = load_json(WEIGHTS)
    survived = survived_dict(train)
    baseline = baseline_dict(train)
    survived_test = survived_dict(test)
    start_loss = calculate_mean_loss(train, survived, weights, features_list, age_values)
    train_context = {"weights": weights, 
                     "survived": survived,
                     "start_loss": start_loss,
                     "survived_test": survived_test,
                     "features_list": features_list,
                     "age_values": age_values,
                     "baseline": baseline}
    return train_context


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
                                          actual = survived,
                                          weights = weights,
                                          steps = STEPS_FOR_TRAIN,
                                          iters = NUMBER_OF_ITERATIONS,
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
            start_loss = training(test,train,seed)
            impact_weights(start_loss,seed)
            
def training(test: list, 
             train: list, 
             seed: int, 
             is_show_progress: bool = False, 
             is_show_result: bool = False) -> dict:
    """Обучение, обновыление весов и вывод losss"""
    
    train_context = init_train_dicts(train,test)
    weights_train, time_train = train_classifier(raw_list = train,
                                              actual = train_context["survived"],
                                              weights = train_context["weights"],
                                              steps = STEPS_FOR_TRAIN,
                                              iters = NUMBER_OF_ITERATIONS,
                                              seed_value = seed,
                                              features_list = train_context["features_list"],
                                              age_values = train_context["age_values"],
                                              show_progress = is_show_progress)
    
    loss_train = calculate_mean_loss(test,train_context["survived_test"],weights_train,train_context["features_list"], train_context["age_values"])
    if is_show_result is True:
        show_result(train = train,
                    test = test,
                    seed = seed,
                    weights = train_context["weights"],
                    new_weights = weights_train,
                    baseline = train_context["baseline"],
                    survived = train_context["survived"],
                    survived_test = train_context["survived_test"],
                    features_list = train_context["features_list"],
                    age_values = train_context["age_values"],
                    start_loss = train_context["start_loss"],
                    new_loss = loss_train,
                    time_train = time_train
                    )
    return {
        "weights_train": weights_train,
        "loss_train": loss_train,
        "time_train": time_train
    }
    
def main_func():
    train, test  = get_train_csv_lists(TRAIN, seed_value = SEED_SPLIT)
    train, test = replace_median_ages(train), replace_median_ages(test)
    res_train = training(test, train, SEED_TRAIN, True, True)
    save_json(WEIGHTS, res_train["weights_train"])
    
if __name__ == "__main__":
    os.system("cls")
    main_func()
    

