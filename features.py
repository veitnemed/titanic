

def get_column_in_matrix(data: list, feature: str) -> list:
    "Получаем список значений столбца feature (НЕ СТРОКА) "
    return [passenger[feature] for passenger in data]

def median_age(data: list) -> float:
    ages = [x for x in get_column_in_matrix(data, "Age") if x != ""]
    l = len(ages)
    ages_sort = sorted(list(map(float,ages)))
    if l % 2 == 1:
        return ages_sort[int(l/2)]
    else:
        return (ages_sort[int(l/2)] + ages_sort[(int(l/2)-1)])/2
    
#from storage import get_full_csv_list
#from config import TRAIN
#l = get_full_csv_list(TRAIN)
#print(median_age(l))

def replace_median_ages(data: list):
    
    median = median_age(data)
    for obj in data:
        if obj["Age"] == "":
            obj["Age"] = str(median)
    return data
        
def replace_feature_values(data: list, feature, replace_dict: dict):
    for p in data:
        p[feature] = replace_dict[p[feature]]
    return data

