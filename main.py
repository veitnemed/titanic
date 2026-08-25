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
    
def show_result_info(binare_dict: dict, actual_survived: dict, mess: str, time_proccessing = None):
    print("="*50)
    lenth = len(binare_dict)
    n = number_of_prediction(binare_dict, actual_survived)
    p = procent_prediction(lenth, n)
    print(f"{mess}: {n} / {lenth} ({p} % правильных)")
    if time_proccessing != None:
        print(f"\nВремя обучения {time_proccessing} сек.") 

def result(raw_dict: dict, wights: dict, actual_survived, mes: str, time_procesing = None):
    if isinstance(raw_dict,list):
        new_score_dict = create_dict_scores(raw_dict, wights)
        new_binare_dict = create_dict_binare(new_score_dict, THREASHOLD)
    else:
        new_binare_dict = raw_dict
    show_result_info(new_binare_dict, actual_survived, mes, time_procesing)
    
def main_func():
    """Главная функция преокта"""
    
    raw_dict_train, raw_dict_test  = get_train_csv_lists(TRAIN)
    actual_survived = survived_dict(raw_dict_train)
    baseline = baseline_dict(TRAIN)
    actual_survived_test = survived_dict(raw_dict_test)
    
    result(raw_dict = baseline,
           wights = first_weights,
           actual_survived = actual_survived,
           mes = "Бейслайн")
    
    result(raw_dict = raw_dict_train,
           wights = first_weights,
           actual_survived = actual_survived,
           mes = "До обучения")

    new_weight, time_train = train_classifier(raw_dict = raw_dict_train,
                                  actual_survived = actual_survived,
                                  weights = first_weights,
                                  steps = STEPS_FOR_TRAIN,
                                  iters = NUMBER_OF_ITERATIONS,
                                  treashold = THREASHOLD)
    result( raw_dict = raw_dict_train,
            wights = new_weight,
            actual_survived = actual_survived,
            mes = "После обучения весов",
            time_procesing = time_train)
    
    result( raw_dict = raw_dict_test,
            wights = new_weight,
            actual_survived = actual_survived_test,
            mes = "Тестовый набор")
    
if __name__ == "__main__":
    main_func()

