from storage import (get_full_csv_list, 
                     save_csv,
                     survived_dict,
                     baseline_dict, get_train_csv_lists)
from config import  (TRAIN, 
                     COLUMNS_BINARE, 
                     RESULT, 
                     first_weights,
                     STEPS_FOR_TRAIN,
                     NUMBER_OF_ITERATIONS,
                     THREASHOLD)

from scores import (
                    survived_counter, 
                    number_of_prediction,
                    procent_prediction,
                    mean_csv_result, create_dict_scores, create_dict_binare)

from model import predict_dataset, evaluate_weights, train_classifier, best_threshold

def show_main_info_for_traning(score_dict: dict, binare_dict: dict, mean_score: float):
    """Выводится основная информация о прогнозе"""
    print(f"Threshold: {round(mean_score,2)}")
    print("Min score:", round(min(list(score_dict.values())),2))
    print("Max score:", round(max(list(score_dict.values())),2))
    print(f"Количество выживших по предсказанию: {survived_counter(binare_dict)}")
    
def show_result_info(binare_dict: dict, actual_survived: dict, mess: str):
    
    lenth = len(binare_dict)
    n = number_of_prediction(binare_dict, actual_survived)
    p = procent_prediction(lenth, n)
    print(f"{mess}: {n} / {lenth} ({p} % правильных)")
    

def main_func():
    """Главная функция преокта"""
    raw_dict_train, raw_dict_test  = get_train_csv_lists(TRAIN)
    
    score_dict3 = create_dict_scores(raw_dict_train, first_weights)
    binare_dict3 = create_dict_binare(score_dict3, THREASHOLD)
    #save_csv(binare_dict, COLUMNS_BINARE, RESULT)
    actual_survived3 = survived_dict(raw_dict_train)
    
    baseline = baseline_dict(TRAIN)
    show_result_info(baseline, actual_survived3, "\nBaseline")
    print("="*50)
    show_result_info(binare_dict3, actual_survived3, "До обучения")
    print("="*50)
    
    new_weight, time_train = train_classifier(raw_dict = raw_dict_train,
                                  actual_survived = actual_survived3,
                                  weights = first_weights,
                                  steps = STEPS_FOR_TRAIN,
                                  iters = NUMBER_OF_ITERATIONS,
                                  treashold = THREASHOLD)
    new_score_dict = create_dict_scores(raw_dict_train,new_weight)
    new_binare_dict1 = create_dict_binare(new_score_dict, THREASHOLD)
    show_result_info(new_binare_dict1, actual_survived3, "После обучения весов")
    print(f"Время обучения {time_train} сек.")
    
    print("="*50)
    score_dict3 = create_dict_scores(raw_dict_test, new_weight)
    binare_dict3 = create_dict_binare(score_dict3, THREASHOLD)
    actual_survived3 = survived_dict(raw_dict_test)
    show_result_info(binare_dict3, actual_survived3, f"Тестовый набор ({len(raw_dict_test)} пас.)")   
if __name__ == "__main__":
    main_func()

