from storage import (get_csv_dict, 
                     save_csv,
                     survived_dict,
                     baseline_dict)
from config import  (TRAIN, COLUMNS_BINARE, RESULT,
                               
                                )

from scores import (create_dict_scores, 
                    create_dict_binare, 
                    survived_counter, 
                    mean_csv_result,
                    number_of_prediction,
                    procent_prediction,)

def show_main_info_for_traning(score_dict: dict, binare_dict: dict, mean_score: float):
    """Выводится основная информация о прогнозе"""
    print(f"Threshold: {round(mean_score,2)}")
    print("Min score:", round(min(list(score_dict.values())),2))
    print("Max score:", round(max(list(score_dict.values())),2))
    print(f"Количество выживших по предсказанию: {survived_counter(binare_dict)}")
    

    
    pass
def show_result_info(binare_dict: dict, actual_survived: dict, mess: str):
    print(f"{mess}\n")
    lenth = len(binare_dict)
    n = number_of_prediction(binare_dict, actual_survived)
    p = procent_prediction(lenth, n)
    print(f"Количество правильных предсказаний: {n} из {lenth}")
    print(f"{p} % правильных, {round(100-p,2)} % ошибок")



def init_dicts(name_dataset: str, name_binare: str)-> tuple:
    """Инциализация стартовых словарей"""
    raw_dict = get_csv_dict(name_dataset)
    score_dict = create_dict_scores(raw_dict)
    threshold = mean_csv_result(score_dict)
    binare_dict = create_dict_binare(score_dict, threshold)
    save_csv(binare_dict, COLUMNS_BINARE, name_binare)
    return (raw_dict, score_dict, binare_dict, threshold)

def main_func():
    raw_dict, score_dict, binare_dict, threshold = init_dicts(TRAIN, RESULT)
    actual_survived = survived_dict(TRAIN)
    
    show_result_info(binare_dict, actual_survived, "Тренировочный датасет")
    show_main_info_for_traning(score_dict, binare_dict, threshold)
    print(baseline_dict(TRAIN))
        
if __name__ == "__main__":
    main_func()

