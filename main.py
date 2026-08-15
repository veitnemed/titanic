import csv
from storage import get_csv_dict, save_csv_predict, load_scores
from config import first_weights, SCORES_CSV, TRAIN_CSV, FEATURES

def get_score(passenger: dict) -> float:
    """Считаем итоговое значение для одного пассажира"""
    score = 0.0
    for feature in FEATURES:
        value = passenger[feature]
        if value != "":
            score+= float(first_weights[feature][value])
    return score

def create_dict_scores(dataset: dict) -> dict:
    """Собирает словарь с предиктом для каждого пасажира"""
    result = {}
    for passenger in dataset:
        result[passenger["PassengerId"]] = get_score(passenger)
    return result

def survived_counter(file_name: str) -> int:
    """Считает количекство выживших в датесете file_name"""
    with open(file_name,'r',encoding="utf-8", newline="") as f:
        csv_list = list(csv.DictReader(f))
    counter = 0
    for obj in csv_list:
        counter += int(obj["Survived"])
    return counter

def main_func():
    data = get_csv_dict(TRAIN_CSV)
    score_dict = create_dict_scores(data)
    save_csv_predict(score_dict)

if __name__ == "__main__":
    main_func()
