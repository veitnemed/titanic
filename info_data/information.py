"Вывод основной информации о датасетах"

import csv

from funcs_for_information import (all_status, 
                                   survived_in_group, 
                                   group_year_range, 
                                   empty_value_counter)
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# __file__ - текущий файл (information.py)
# os.path.dirname(__file__) - путь к папке
# >> c:\Users\super\Desktop\vscode projects\titanic\info_data


    
from config import (ru_column,  
                    TRAIN,
                    DATA_TEST)



with open(DATA_TEST, 'r', encoding='utf-8') as f1:
    dict_test = list(csv.DictReader(f1))
with open(TRAIN, 'r', encoding='utf-8') as f1:
    dict_train = list(csv.DictReader(f1))
os.system("cls")
features = list(dict_test[0].keys())

print("======== ИНФОРМАЦИЯ О ФАЙЛАХ CSV =========\n")
print(f"Количество строк в test.csv: {len(dict_test)}")
print(f"Количество строк в train.csv: {len(dict_train)}")
print(f"Количество заголвоков столбцов в test.csv: {len(dict_test[0].keys())}")
print(f"Количество заголвоков столбцов в train.csv: {len(dict_train[0].keys())}")
print(f"Заголовки в test.csv : {sorted(list(dict_test[0].keys()))}")   
print(f"Заголовки в train.csv: {sorted(list(dict_train[0].keys()))}")
print(f"Равенство заголвоков столбцов в test.csv и train.csv: {sorted(list(dict_test[0].keys())) == sorted(list(dict_train[0].keys()))}\n")
print(f"Перевод на русский: {list(ru_column.values())}")


def get_procent(part, all):
    return round(100*(part/all),1)


number_of_male, number_of_survived_male = survived_in_group(dict_train, "Sex", "male")
number_of_female, number_of_survived_female = survived_in_group(dict_train, "Sex", "female")
number_of_people, number_of_survived = len(dict_train), number_of_survived_male + number_of_survived_female
print("\n\n======== ПУСТЫЕ ПОЛЯ (train.csv) =========\n")
for feature in features:
    print(f"Признак {feature}: пустых значений {empty_value_counter(dict_train, feature)} ")


print("\n\n======== ПРИЗНАК: ПОЛ (train.csv) =========\n")

print(f"Всего пассажиров в датасете: {number_of_people}")
print(f"Количество мужчин: {number_of_male} ({get_procent(number_of_male, number_of_people)} %)")
print(f"Количество женщин : {number_of_female} ({get_procent(number_of_female, number_of_people)} %)")
print(f" Доля выживших пассажиров: {get_procent(number_of_survived, number_of_people)}\n")

print(f"Общее количество выживших: {number_of_survived} ")
print(f"Колчиество выживших мужчин {number_of_survived_male} ({get_procent(number_of_survived_male, number_of_survived)} %)")
print(f"Колчиество выживших женщин {number_of_survived_female} ({get_procent(number_of_survived_female, number_of_survived)} %)")


print("\n\n======== ЗНАЧЕНИЯ ПРИЗНАКОВ (train.csv) =========\n")

for column in dict_train[0].keys():
    status_set = all_status(dict_train, column)
    lenth = len(status_set)
    print(f"Уникальных значениий признака '{column} // {ru_column.get(column, column)} - {lenth}")
    print(f"Первые 10 значений:")
    print(*sorted(status_set)[:10],"\n")

print("\n\n======== ПРИЗНАК: КЛАСС ПАССАЖИРА (train.csv) =========\n")
pclass = ["1", "2", "3"]

for x in pclass:
    count_q, count_q_sur = survived_in_group(dict_train, 'Pclass', x)
    print(f"Из класса пассижиров {x} ({count_q} человек) выижило  {count_q_sur} // {get_procent(count_q_sur, count_q)} %")

print("\n======== ПРИЗНАК: ПОРТ ПОСАДКИ (train.csv) =========\n")
groups = ["Q","S","C"]
for x in groups:
    count_q, count_q_sur = survived_in_group(dict_train, 'Embarked', x)
    print(f"Из порта {x} ({count_q} человек) выижило {count_q_sur} // {get_procent(count_q_sur, count_q)} %")



print("\n\n======== ПРИЗНАК: Количество супругов/сестер/братьев/сестер пассажира на борту  =========\n")

amounts_sibsb = ["0","1","2","3","4","5","8"] 
for x in amounts_sibsb:
    count_q, count_q_sur = survived_in_group(dict_train, "SibSp", x)
    if count_q != 0:
        status_set = count_q_sur/count_q*100
    else:
        status_set = -1
    print(f"Из группы {x} ({count_q} человек) выижило {get_procent(count_q_sur, count_q)} %")


print("\n\n======== ПРИЗНАК: Количество родителей/детей пассажира на борту  =========\n")

parch = ["0","1","2","3","4","5","6"] 
for x in parch:
    count_q, count_q_sur = survived_in_group(dict_train, "Parch", x)
    if count_q != 0:
        status_set = count_q_sur/count_q*100
    else:
        status_set = -1
    print(f"Из группы {x} ({count_q} человек) выижило {get_procent(count_q_sur, count_q)} %")


print("\n\n======== РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ  =========\n")
step = 10
for k, v in sorted(group_year_range(dict_train, step).items()):
    if k == -1:
        print(f"Неизвестен возраст: {v}")
    else:

        print(f"От {int(k)} до {int(k+step)}: {v}")

print("====================================")
print("====================================")
print("====================================")
print("\n\n======== TRAIN_CSV  =========\n")


print("\n\n======== ЗНАЧЕНИЯ ПРИЗНАКОВ (test.csv) =========\n")

for column in dict_train[0].keys():
    if column != "Survived":
        status_set = all_status(dict_test, column)
        lenth = len(status_set)
        print(f"Уникальных значениий признака '{column} // {ru_column.get(column, column)} - {lenth}")
        print(f"Первые 10 значений:")
        print(*sorted(status_set)[:10],"\n")

