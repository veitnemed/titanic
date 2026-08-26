"Глобальные константы, "

RESULT = "C:/Users/super/Desktop/vscode projects/titanic/output_data/result_train_binare.csv"
TRAIN  = 'C:/Users/super/Desktop/vscode projects/titanic/datasets/train.csv'


CORRELATION_FEATURES = ["Age","Pclass","Sex","Parch","SibSp","Fare","Survived"]

COLUMNS_PREDICT = ["PassengerId", "Score"]
COLUMNS_BINARE = ["PassengerId", "Survived"]

STEPS_FOR_TRAIN = [1, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025]
NUMBER_OF_ITERATIONS = 500
THREASHOLD = 0.5

first_weights = {
    "Sex": {
        "female": 1,
        "male": 1,
    },
    "SibSp": {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "8": 1

    },
    "Parch": {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "6": 1,
    },
    "Pclass": {
        "1": 1,
        "2": 1,
        "3": 1
    },
    "Embarked": {
        "Q": 1,
        "S": 1,
        "C": 1
    }

}



FEATURES = list(first_weights.keys())

ru_column = {
    "Age": "Возраст пассажира",
    "Cabin": "Каюта",
    "Embarked": "Порт посадки",
    "Fare": "Стоимость билета",
    "Name": "Имя пассажира",
    "Parch": "Количество родителей/детей пассажира на борту",
    "PassengerId": "ID пассажира",
    "Pclass": "Класс пассажира",
    "Sex": "Пол пассажира",
    "SibSp": "Количество супругов/сестер/братьев/сестер пассажира на борту",
    "Ticket": "Номер билета"
}