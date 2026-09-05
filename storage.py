"Модуль отвечает за работу с файлами"

import csv
import os 
import json 
    
    
def get_full_csv_list(file_name) -> list[dict]:
    with open(file_name, 'r', encoding='utf-8') as f: 
        data = list(csv.DictReader(f))
    return data

def get_train_csv_lists(filename: str, seed_value = 1) -> tuple[list, list]:
    """Рандомно разделяет датасет 80/20"""
    import random as rnd
    rng = rnd.Random(seed_value)
    
    full_data = get_full_csv_list(filename)
    train_data = full_data.copy()
    
    len_dataset = len(full_data)
    len_train = int((4/5)*len_dataset)
    len_test = len_dataset - len_train
    test_data = []
    
    for _ in range(len_test):
        obj = rng.choice(train_data)
        train_data.remove(obj)
        test_data.append(obj)
    
    return train_data , test_data
        
    
    
def save_csv(d: dict, columns: list, filename: str):
    """Сохранение csv предикта """
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        writer.writeheader()

        for passenger_id, score in d.items():
            
            writer.writerow({
                columns[0]: passenger_id,
                columns[1]: score
            })
            
def load_scores(filename: str) -> dict:
    scores = {}

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scores[row["PassengerId"]] = float(row["Score"])

    return scores

def survived_dict(data: dict) -> dict:
    """Создаём словарь типа id -> survived"""
    result = {}
    for idx, d in enumerate(data):
        result[d["PassengerId"]] = int(d["Survived"])
    return result 

def baseline_dict(csv_list):
    """Создается словарь по логике если пол =  М то 0, иначе 1"""
    baseline = {}
    
    for passenger in csv_list:
        if passenger["Sex"] == "male":
            baseline[passenger["PassengerId"]] = 0
        elif passenger["Sex"] == "female":
            baseline[passenger["PassengerId"]] = 1
            
    return baseline

def file_exists(filename: str):
    """Проверка существует ли файл"""
    return os.path.exists(filename)

def empty_file(filename: str):
    with open(filename, 'r', encoding="utf-8") as f:
            weights = json.load(f)
    return len(weights) == 0

def create_json(filename, default_weights):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(default_weights, f) 

def clean_json(filename):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(dict(), f) 

def load_json(filename: str):
    with open(filename, 'r', encoding="utf-8") as f:
        weights = json.load(f)
    return weights

def save_json(filename: str, weights: dict):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(weights, f, indent=3)

def json_init(filename: str, default_weights: dict):
    """
    1. Если папки нет, создаем 
    2. Если фй
    """
    directory = os.path.dirname(filename)
    if file_exists(directory) is False:
        os.makedirs(directory)
    if file_exists(filename) is False:
        create_json(filename, default_weights)
    if empty_file(filename) is True:
        create_json(filename, default_weights)


