"Вывод основной информации о датасетах"

import csv

from funcs_for_information import (all_status, 
                                   survived_in_group, 
                                   group_year_range, 
                                   empty_value_counter, survived_in_range)
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from features import replace_median_ages
# __file__ - текущий файл (information.py)
# os.path.dirname(__file__) - путь к папке
# >> c:\Users\super\Desktop\vscode projects\titanic\info_data


    
from config import (ru_column,  
                    TRAIN)




with open(TRAIN, 'r', encoding='utf-8') as f1:
    dict_train = list(csv.DictReader(f1))
os.system("cls")
dict_train = replace_median_ages(dict_train)
features = list(dict_train[0].keys())

print("======== ИНФОРМАЦИЯ О ФАЙЛАХ CSV =========\n")

print(f"Количество строк в train.csv: {len(dict_train)}")

print(f"Количество заголвоков столбцов в train.csv: {len(dict_train[0].keys())}")
print(f"Заголовки в train.csv: {sorted(list(dict_train[0].keys()))}")
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
print(f"Колчиество выживших мужчин {number_of_survived_male} ({get_procent(number_of_survived_male, number_of_male)} %)")
print(f"Колчиество выживших женщин {number_of_survived_female} ({get_procent(number_of_survived_female, number_of_female)} %)")


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
step = 20
for k, v in sorted(group_year_range(dict_train, step).items()):
    if k == -1:
        print(f"Неизвестен возраст: {v}")
    else:

        print(f"От {int(k)} до {int(k+step)}: {v}")


print("\n\n======== ВЫЖИВАЕМОСТЬ ПО ВОЗРАСТУ  =========\n")

ages = [20,100]


counter_ages, survived_ages = survived_in_range(dict_train,ages)

for group, amount in sorted(counter_ages.items()):
    p = round(100*survived_ages[group]/amount,2)
    print(f"Группа {group}: {amount} чел. // Выжило: {p} %")