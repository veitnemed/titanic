"Глобальные константы, "
import os 
dirname = os.path.dirname(__file__)
TRAIN = os.path.join(dirname, "datasets/train.csv")
RESULT = os.path.join(dirname, "output_data/result_train_binare.csv")
WEIGHTS = os.path.join(dirname,"weights.json")

STEPS_FOR_TRAIN = [1, 0.5, 0.25, 0.125, 0.05, 0.01]
NUMBER_OF_ITERATIONS = 100
THREASHOLD = 0.5
SEED_SPLIT = 43
SEED_TRAIN = SEED_SPLIT

CORRELATION_FEATURES = ["Age","Pclass","Sex","Parch","SibSp","Fare","Survived", "Loss"]

COLUMNS_PREDICT = ["PassengerId", "Score"]
COLUMNS_BINARE = ["PassengerId", "Survived"]

DEFAULT_WEIGHTS = {
    "Sex": {
        "female": 0,
        "male": 0,
    },
    "SibSp": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "8": 0

    },
    "Parch": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
    },
    "Pclass": {
        "1": 0,
        "2": 0,
        "3": 0
    },
    "Embarked": {
        "Q": 0,
        "S": 0,
        "C": 0
    },

        "Male & zero": {
      "Male & zero": 0
   },
    "Bias": {"bias": 0}

}
#del DEFAULT_WEIGHTS["Male & zero"]

FEATURES = list(DEFAULT_WEIGHTS.keys())

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