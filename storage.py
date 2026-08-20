"Модуль отвечает за работу с файлами"

import csv

def get_csv_dict(file_name):
    with open(file_name, 'r', encoding='utf-8') as f: 
        data = list(csv.DictReader(f))
    return data

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

def survived_dict(filename:str) -> dict:
    """Создаём словарь типа id -> survived"""
    result = {}
    data = get_csv_dict(filename)
    for idx, d in enumerate(data):
        result[d["PassengerId"]] = int(d["Survived"])
    return result 

def baseline_dict(filename):
    """Создается словарь по логике если пол =  М то 0, иначе 1"""
    csv_list = get_csv_dict(filename)
    baseline = {}
    
    for passenger in csv_list:
        if passenger["Sex"] == "male":
            baseline[passenger["PassengerId"]] = 0
        elif passenger["Sex"] == "female":
            baseline["PassengerId"] = 1
            
    return baseline



