SCORES_CSV = "result.csv"
TRAIN_CSV = 'train.csv'

first_weights = {
    "Sex": {
        "female": 0.68,
        "male": 0.32
    },
    "SibSp": {
        "0": 0.345,
        "1": 0.536,
        "2": 0.464,
        "3": 0.25,
        "4": 0.167,
        "5": 0,
        "8": 0

    },
    "Parch": {
        "0": 0.341,
        "1": 0.551,
        "2": 0.5,
        "3": 0.6,
        "4": 0,
        "5": 0.2,
        "6": 0
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