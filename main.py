from storage import (
                     survived_dict,
                     baseline_dict, get_train_csv_lists)
from config import  (TRAIN, 
                     first_weights,
                     STEPS_FOR_TRAIN,
                     NUMBER_OF_ITERATIONS,
                     THREASHOLD)

from scores import (
                    survived_counter, 
                    number_of_prediction,
                    percent_prediction,
                    create_dict_scores, create_dict_binary, scores_to_sigmoids)

from model import train_classifier, calculate_mean_loss
import os

def show_main_info_for_training(scores: dict, binary: dict, mean_score: float):
    """Выводится основная информация о прогнозе"""
    print(f"Threshold: {round(mean_score,2)}")
    print("Min score:", round(min(list(scores.values())),2))
    print("Max score:", round(max(list(scores.values())),2))
    print(f"Количество выживших по предсказанию: {survived_counter(binary)}")
    
def show_result_info(binary: dict, 
                     survived: dict, 
                     message: str, 
                     time_processing = None):
    print("="*50)
    length = len(binary)
    n = number_of_prediction(binary, survived)
    percent = percent_prediction(length, n)
    print(f"{message}: {n} / {length} ({percent} % правильных)")
    if time_processing != None:
        print(f"\nВремя обучения {time_processing} сек.") 

def result(raw_list: list, 
           weights: dict, 
           survived, 
           message: str, 
           time_processing = None):
    scores = create_dict_scores(raw_list, weights)
    binary = create_dict_binary(scores_to_sigmoids(scores), THREASHOLD)
    show_result_info(binary, survived, message, time_processing)
    maean_loss = calculate_mean_loss(raw_list, survived, weights)
    print(f"Mean loss: {round(maean_loss,2)}")
    
def show_weights(weights, new_weights):
    
    for feature, val in weights.items():
        print(f"{feature}: ")
        for k, v in val.items():
            v1 = new_weights[feature][k]
            if k == "max_key":
                continue
            if abs(v-v1) < 0.0001:
                print(f"{k}: {v} (без изменений)")
                continue
            print(f"{k}: {v} -> {round(v1,2)}")
    
    
def main_func():
    """Главная функция преокта"""
    
    train, test  = get_train_csv_lists(TRAIN)
    survived = survived_dict(train)
    baseline = baseline_dict(train)
    survived_test = survived_dict(test)
    
    show_result_info(binary = baseline,
                    survived = survived,
                    message = "Baseline")
    result(raw_list = train,
           weights = first_weights,
           survived = survived,
           message = "До обучения")
    new_weights, time_train = train_classifier(raw_list = train,
                                  actual = survived,
                                  weights = first_weights,
                                  steps = STEPS_FOR_TRAIN,
                                  iters = NUMBER_OF_ITERATIONS)
    result(raw_list = train,
           weights = new_weights,
           survived = survived,
           message = "После обучения весов",
           time_processing = time_train)
    result(raw_list = test,
           weights = new_weights,
           survived = survived_test,
           message = "Тестовый набор")
    
    show_weights(first_weights, new_weights)
    
if __name__ == "__main__":
    os.system("cls")
    main_func()

