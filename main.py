from storage import (get_csv_dict, 
                     save_csv,
                     survived_dict,
                     baseline_dict)
from config import  (TRAIN, COLUMNS_BINARE, RESULT, first_weights)

from scores import (
                    survived_counter, 
                    number_of_prediction,
                    procent_prediction,)

from model import predict_dataset, evaluate_weights, train_classifier

def show_main_info_for_traning(score_dict: dict, binare_dict: dict, mean_score: float):
    """Выводится основная информация о прогнозе"""
    print(f"Threshold: {round(mean_score,2)}")
    print("Min score:", round(min(list(score_dict.values())),2))
    print("Max score:", round(max(list(score_dict.values())),2))
    print(f"Количество выживших по предсказанию: {survived_counter(binare_dict)}")
    
def show_result_info(binare_dict: dict, actual_survived: dict, mess: str):
    print(f"{mess}\n")
    lenth = len(binare_dict)
    n = number_of_prediction(binare_dict, actual_survived)
    p = procent_prediction(lenth, n)
    print(f"Количество правильных предсказаний: {n} из {lenth}")
    print(f"{p} % правильных, {round(100-p,2)} % ошибок")

def show_result_tarino(weights, new_weights, binare_dict: dict, actual_survived: dict, mess: str):
    print(f"{mess}\n")
    lenth = len(binare_dict)
    n_new = number_of_prediction(binare_dict, actual_survived)
    p = procent_prediction(lenth, n)
    
    print(f"Количество правильных предсказаний: {n} из {lenth}")
    print(f"{p} % правильных, {round(100-p,2)} % ошибок")

def main_func():
    raw_dict = get_csv_dict(TRAIN)
    score_dict, binare_dict, threshold = predict_dataset(raw_dict, first_weights)
    save_csv(binare_dict, COLUMNS_BINARE, RESULT)
    actual_survived = survived_dict(TRAIN)
    
    show_result_info(binare_dict, actual_survived, "Default classifie:")
    show_main_info_for_traning(score_dict, binare_dict, threshold)
    print("="*50)
    print("="*50)
    baseline = baseline_dict(TRAIN)
    show_result_info(baseline, actual_survived, "\n\n\nBaseline: ")
    print("="*50)
    print("="*50)
    step = 0.5
    iters = 10000
    new_weight = train_classifier(raw_dict, actual_survived, first_weights, step, iters)
    score_dict1, binare_dict1, threshold1 = predict_dataset(raw_dict, new_weight)
    show_main_info_for_traning(score_dict1, binare_dict1, threshold1)
    show_result_info(binare_dict1, actual_survived, "\n\n\nResult train:")
    print("="*50)
    print("="*50,"\n\n\n")
    print("Weights:")
    
        
    
    
if __name__ == "__main__":
    main_func()

