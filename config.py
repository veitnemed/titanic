"Глобальные константы, "

RESULT= "C:/Users/super/Desktop/vscode projects/titanic/output_data/result_train_binare.csv"
TRAIN = 'C:/Users/super/Desktop/vscode projects/titanic/datasets/train.csv'


CORRELATION_FEATURES = ["Age","Pclass","Sex","Parch","SibSp","Fare","Survived"]

COLUMNS_PREDICT = ["PassengerId", "Score"]
COLUMNS_BINARE = ["PassengerId", "Survived"]

STEPS_FOR_TRAIN = [1, 0.5, 0.25, 0.1, 0.05, 0.001]
NUMBER_OF_ITERATIONS = 1000
THREASHOLD = 0.5

first_weights = {
    "Sex": {
        "female": -2,
        "male": -4.5,
    },
    "SibSp": {
        "0": 0.9,
        "1": 1.2,
        "2": 0.9,
        "3": -0.9,
        "4": 0,
        "5": -1.7,
        "8": -2.6

    },
    "Parch": {
        "0": 3,
        "1": 4,
        "2": 4,
        "3": 5,
        "4": 0.25,
        "5": 1.85,
        "6": 1.24,
    },
    "Pclass": {
        "1": 0,
        "2": -0.6,
        "3": -1.608
    },
    "Embarked": {
        "Q": 0.24,
        "S": -0.313,
        "C": 0
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