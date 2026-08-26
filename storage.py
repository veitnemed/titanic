"Модуль отвечает за работу с файлами"

import csv

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



