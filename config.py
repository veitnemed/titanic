"Глобальные константы, "

RESULT= "C:/Users/super/Desktop/vscode projects/titanic/output_data/result_train_binare.csv"
TRAIN = 'C:/Users/super/Desktop/vscode projects/titanic/datasets/train.csv'


CORRELATION_FEATURES = ["Age","Pclass","Sex","Parch","SibSp","Fare","Survived"]

COLUMNS_PREDICT = ["PassengerId", "Score"]
COLUMNS_BINARE = ["PassengerId", "Survived"]

STEPS_FOR_TRAIN = [0.75, 0.5, 0.25, 0.1, 0.05]
NUMBER_OF_ITERATIONS = 1000
THREASHOLD = 1.91

first_weights = {
    "Sex": {
        "female": 0.742,
        "male": 0.189
    },
    "SibSp": {
        "0": 0.345,
        "1": 0.536,
        "2": 0.464,
        "3": 0.25,
        "4": 0.167,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "max_key": 8

    },
    "Parch": {
        "0": 0.341,
        "1": 0.551,
        "2": 0.5,
        "3": 0.6,
        "4": 0,
        "5": 0.2,
        "6": 0,
        "max_key": 6
    },
    "Pclass": {
        "1": 0.63,
        "2": 0.473,
        "3": 0.242
    },
    "Embarked": {
        "Q": 0.39,
        "S": 0.337,
        "C": 0.554
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