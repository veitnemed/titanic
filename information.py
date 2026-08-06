import csv
import os 

with open('test.csv','r',encoding='utf-8') as f, open('train.csv', 'r', encoding='utf-8') as f1:
    dict_test = list(csv.DictReader(f))
    dict_train = list(csv.DictReader(f1))

os.system("cls")
print("======== ИНФОРМАЦИЯ О ДАТАСЕТАХ =========\n")
print(f"Количество строк в test.csv: {len(dict_test)}")
print(f"Количество строк в train.csv: {len(dict_train)}\n")

print(f"Заголовки столбцов в test.csv: {sorted(list(dict_test[0].keys()))}")   
print(f"Заголовки столбцов в train.csv: {sorted(list(dict_train[0].keys()))}\n")

print(f"Равенство заголвоков столбцов в test.csv и train.csv: {sorted(list(dict_test[0].keys())) == sorted(list(dict_train[0].keys()))}\n")

print(f"Количество заголвоков столбцов в test.csv: {len(dict_test[0].keys())}")
print(f"Количество заголвоков столбцов в train.csv: {len(dict_train[0].keys())}\n")

count_survived = 0 
count_survived_male = 0
count_male = 0

for obj in dict_train:
    count_survived += int(obj["Survived"])
    if obj["Sex"] == "male":
        count_male += 1
    if obj["Sex"] == "male" and obj["Survived"] == "1":
        count_survived_male += 1        
print(f"Количество мужчин в train.csv: {count_male} из {len(dict_train)} ({round(count_male/len(dict_train)*100, 2)}%)\n")
print(f"Количество женщин в train.csv: {len(dict_train) - count_male} из {len(dict_train)} ({round((len(dict_train) - count_male)/len(dict_train)*100, 2)}%)\n")
print(f"Количество выживших в train.csv: {count_survived} из {len(dict_train)} ({round(count_survived/len(dict_train)*100, 2)}%)\n")
print(f"Количество выживших мужчин в train.csv: {count_survived_male} из {count_survived} ({round(count_survived_male/count_survived*100, 2)}%)\n")    
print("Количество выживших женщин в train.csv: ", count_survived - count_survived_male, "из", count_survived, "(", round((count_survived - count_survived_male)/count_survived*100, 2), "%)\n")         


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
