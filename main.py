from storage import (get_csv_dict, 
                     save_csv,
                     survived_dict,
                     baseline_dict,)
from config import  (TRAIN, COLUMNS_BINARE, RESULT, first_weights)

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
    

def show_result_tarino(weights, new_weights, binare_dict: dict, actual_survived: dict, mess: str):
    print(f"{mess}\n")
    lenth = len(binare_dict)
    n_new = number_of_prediction(binare_dict, actual_survived)
    p = procent_prediction(lenth, n)
    
    print(f"Количество правильных предсказаний: {n} из {lenth}")
    print(f"{p} % правильных, {round(100-p,2)} % ошибок")

def main_func():
    raw_dict = get_csv_dict(TRAIN)
    score_dict = create_dict_scores(raw_dict, first_weights)
    threshold = 1
    binare_dict = create_dict_binare(score_dict, threshold)
    save_csv(binare_dict, COLUMNS_BINARE, RESULT)
    actual_survived = survived_dict(TRAIN)
    
    show_result_info(binare_dict, actual_survived, "Default classifier:")
    show_main_info_for_traning(score_dict, binare_dict, threshold)
    print("="*50)
    baseline = baseline_dict(TRAIN)
    show_result_info(baseline, actual_survived, "\n\nBaseline: ")
    print("="*50)
    
    steps = [0.75, 0.5, 0.25, 0.1, 0.05]
    iters = 1000
    
    new_weight = train_classifier( raw_dict, actual_survived, first_weights, steps, iters, threshold)
    score_dict1 = create_dict_scores(raw_dict,new_weight)
    binare_dict1 = create_dict_binare(score_dict1, threshold)
    
    
    
    show_main_info_for_traning(score_dict1, binare_dict1, threshold)
    show_result_info(binare_dict1, actual_survived, "\nResult train weights:")
    print("="*50)
    print("="*50)
    threshold_best = best_threshold(raw_dict, actual_survived, 100,new_weight, threshold)
    new_binare_dict = create_dict_binare(score_dict1,threshold_best)
    show_main_info_for_traning(score_dict1, new_binare_dict, threshold_best)
    show_result_info(new_binare_dict, actual_survived, "\nResult train threshold:")
    
    
        
    
    
if __name__ == "__main__":
    main_func()

