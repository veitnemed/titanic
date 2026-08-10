import csv
import os 
from func import all_status, survived_in_group
from constant import ru_column

with open('test.csv','r',encoding='utf-8') as f, open('train.csv', 'r', encoding='utf-8') as f1:
    dict_test = list(csv.DictReader(f))
    dict_train = list(csv.DictReader(f1))

os.system("cls")
print("======== ИНФОРМАЦИЯ О ДАТАСЕТАХ =========\n")
print(f"Количество строк в test.csv: {len(dict_test)}")
print(f"Количество строк в train.csv: {len(dict_train)}\n")

print(f"Заголовки столбцов в test.csv : {sorted(list(dict_test[0].keys()))}")   
print(f"Заголовки столбцов в train.csv: {sorted(list(dict_train[0].keys()))}\n")
print(f"{list(ru_column.values())}")

print(f"Равенство заголвоков столбцов в test.csv и train.csv: {sorted(list(dict_test[0].keys())) == sorted(list(dict_train[0].keys()))}\n")

print(f"Количество заголвоков столбцов в test.csv: {len(dict_test[0].keys())}")
print(f"Количество заголвоков столбцов в train.csv: {len(dict_train[0].keys())}\n")




count_male, count_male_survived = survived_in_group(dict_train, "Sex", "male")
count_female, count_female_survived = survived_in_group(dict_train, "Sex", "female")

print(f"Количество выживщих в train.csv: {count_male_survived + count_female_survived} из {len(dict_train)} \n")
print("===================")
print(f"\nКоличество мужчин в train.csv: {count_male} из {len(dict_train)}")
print(f"Колчиество выживших мужчин {count_male_survived} из {count_male}")
print("===================")
print(f"\nКоличество женщин в train.csv: {count_female} из {len(dict_train)}")
print(f"Колчиество выживших женщин {count_female_survived} из {count_female}")
print("=================")



#for column in dict_train[0].keys():
    #p = all_status(list(dict_train), column)
    #print(f"Уникальные значения в столбце '{column}/{ru_column.get(column, column)}' в train.csv ({len(p)})): {(p, column)}\n")

'''PassengerId
Pclass
Name
Sexv- п
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
'''
