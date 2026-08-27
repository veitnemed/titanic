from storage import (
                     survived_dict,
                     baseline_dict, 
                     get_train_csv_lists,
                     json_init,
                     save_json,
                     load_json)
from config import  (TRAIN, 
                     DEFAULT_WEIGHTS,
                     STEPS_FOR_TRAIN,
                     NUMBER_OF_ITERATIONS,
                     THREASHOLD,
                     WEIGHTS,
                     SEED_SPLIT)

from scores import (
                    survived_counter, 
                    number_of_prediction,
                    percent_prediction,
                    create_dict_scores, 
                    create_dict_binary, 
                    scores_to_sigmoids,
                    summ_weights)

from model import train_classifier, calculate_mean_loss
import os

def show_main_info_for_training(scores: dict, binary: dict, mean_score: float):
    """Выводится основная информация о прогнозе"""
    print(f"Threshold: {round(mean_score,2)}")
    print("Min score:", round(min(list(scores.values())),2))
    print("Max score:", round(max(list(scores.values())),2))
    print(f"Количество выживших по предсказанию: {survived_counter(binary)}")
    
def row_table_v2(binary: dict, 
                     survived: dict, 
                     message: str, 
                     dataset: str):
    """Выввод для бейслайна"""
    length = len(binary)
    n = number_of_prediction(binary, survived)
    accuracy = round((n / length) * 100, 2)
    print("{:<20} {:<20} {:<20} {:<20}".format(*[message, dataset,f"{accuracy} %","---"]))



def row_table(raw_list: list, 
           weights: dict, 
           survived, 
           message: str, 
           dataset: str):
    """Табличный выводрезльтата"""
    scores = create_dict_scores(raw_list, weights)
    binary = create_dict_binary(scores_to_sigmoids(scores), THREASHOLD)
    length = len(binary)
    n = number_of_prediction(binary, survived)
   
    a = (n / length)*100
    maean_loss = calculate_mean_loss(raw_list, survived, weights)
    print("{:<20} {:<20} {:<20} {:<20}".format(*[message, dataset,f"{round(a,2)} %",round(maean_loss,3)]))
   
    
    
    
def show_weights(weights, new_weights):
    """Вывод весов"""
    print("WEIGHT CHANGES")
    for feature, val in weights.items():
    
        print(f"{feature}: ")
        for k, v in val.items():
            v1 = new_weights[feature][k]
            print(f"{k}: {v} -> {round(v1,2)}")
        print()
            
    
def show_main_info(train,test,seed):
    """Выводит шапку"""
    
    print("TITANIC CLASSIFIER\n")
    print(f"Split seed: {seed}")
    print(f"Train: {len(train)} passengers")
    print(f"Validation: {len(test)} passengers\n\n")
    
def main_func():
    """Главная функция преокта"""

    # 1) Начальные рассчёты
    json_init(WEIGHTS, DEFAULT_WEIGHTS)
    weights = load_json(WEIGHTS)
    train, test  = get_train_csv_lists(TRAIN, seed_value = SEED_SPLIT)
    survived = survived_dict(train)
    baseline = baseline_dict(train)
    survived_test = survived_dict(test)
    start_loss = calculate_mean_loss(train,survived, weights )
    
    # 2) Вывод шапки
    show_main_info(train, test, SEED_SPLIT)
    
    # 3) Обучение
    print("TRAINING")
    new_weights, time_train = train_classifier(raw_list = train,
                                      actual = survived,
                                      weights = weights,
                                      steps = STEPS_FOR_TRAIN,
                                      iters = NUMBER_OF_ITERATIONS)
    
    # 4) Результаты 
    print("\n\nRESULTS")
    print("{:<20} {:<20} {:<20} {:<20}".format(*["Model","Dataset","Accuracy ","Loss"]))
    row_table_v2(binary = baseline,
                    survived = survived,
                    message = "Baseline",
                    dataset = "Train")
  
    row_table(raw_list = train,
           weights = weights,
           survived = survived,
           message = "Before training",
           dataset = "Train")
    
    row_table(raw_list = train,
           weights = new_weights,
           survived = survived,
           message = "After training",
           dataset = "Train"
           )
    row_table(raw_list = test,
           weights = new_weights,
           survived = survived_test,
           message = "After training",
           dataset="Validation")
    save_json(WEIGHTS, new_weights)
    
    # 5) Время рассчёта и апдейт
    print(f"\nTraining time: {time_train} sec")
    new_loss = calculate_mean_loss(train,survived,new_weights)
    if new_loss < start_loss:
        w = "updated"
    else:
        w = "not updated"
    print(f"Checkpoint: {w}\n\n")
    #show_weights(weights, new_weights)
    
if __name__ == "__main__":
    os.system("cls")
    main_func()
    

