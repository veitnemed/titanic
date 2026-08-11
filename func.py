
def all_status(train_dataset: list, feature: str) -> set:

    values_set = set()
    for passenger in train_dataset:
        values_set.add(passenger[feature])
    return values_set

def survived_in_group(train_dataset: list, feature: str, value: str) -> tuple:
    """Функция возврщает колчиество выживших, которые соотвествуют признаку feature"""

    count_passengers, count_survived_passengers = 0, 0
    
    for passenger in train_dataset:
        if  passenger[feature] == value:
            count_passengers +=1 
            if  passenger['Survived'] == "1":
                count_survived_passengers += 1
                
    return (count_passengers, count_survived_passengers)

def group_year_range(train_dataset, step_year=10) -> dict:
    """Функция подсчитывает количество человек в разныъ возрастных группах с шагом step_year
        0 - 10 
        10-20
        20 - 30 ... """
    
    from collections import defaultdict
    stat_dict = defaultdict(int)

    for passenger in train_dataset:
        age_str = passenger["Age"]

        if age_str != "":
            age = float(age_str)
            range_year = (age//step_year)*step_year
            stat_dict[range_year] +=1
        else:
            stat_dict[-1] +=1

    return stat_dict

def empty_value_counter(train_dataset: list, feature: str) -> int:
    """Подсчёт количества пустых значений для призанка feature по всем пассажирам"""

    result = 0
    for passenger in train_dataset:
        if passenger[feature] == "":
            result +=1 

    return result
