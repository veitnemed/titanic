import csv

FILE_NAME = "result.csv"
with open('train.csv', 'r', encoding='utf-8') as f:
    data = list(csv.DictReader(f))


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

def get_predict(passenger: dict) -> float:
    score = 0.0
    for feature in FEATURES:
        
            
        value = passenger[feature]
        if value != "":
            score+= float(first_weights[feature][value])
    return score

def all_prediction():
    result = {}
    for passenger in data:
        result[passenger["PassengerId"]] = get_predict(passenger)
    return result

def save_result(d: dict):
    columns = ["PassengerId", "Score"]

    with open(FILE_NAME, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        writer.writeheader()

        for passenger_id, score in d.items():
            writer.writerow({
                "PassengerId": passenger_id,
                "Score": score
            })


def load_scores(filename: str) -> dict:
    scores = {}

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scores[row["PassengerId"]] = float(row["Score"])

    return scores


def get_average_score(scores: dict) -> float:
    total = 0

    for score in scores.values():
        total += float(score)

    return total / len(scores)


def create_submission(scores: dict, threshold: float, filename: str):
    columns = ["PassengerId", "Survived"]

    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for passenger_id, score in scores.items():

            if float(score) > threshold:
                survived = 1
            else:
                survived = 0

            writer.writerow({
                "PassengerId": passenger_id,
                "Survived": survived
            })

def get_sur_amount():
    with open("submission.csv",'r',encoding="utf-8", newline="") as f:
        csv_list = list(csv.DictReader(f))
    c = 0
    for obj in csv_list:
        c += int(obj["Survived"])
    return c

def get_correct_predictions(train_filename: str, prediction_filename: str) -> int:
    predictions = {}

    with open(prediction_filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            predictions[row["PassengerId"]] = row["Survived"]

    correct = 0

    with open(train_filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for passenger in reader:
            passenger_id = passenger["PassengerId"]
            real_survived = passenger["Survived"]

            if predictions[passenger_id] == real_survived:
                correct += 1

    return correct

def main_func():
    score_dict = all_prediction()

    save_result(score_dict)

    scores = load_scores("result.csv")

    threshold = get_average_score(scores)

    print("Средний score:", threshold)

    create_submission(
        scores,
        threshold,
        "submission.csv"
    )
    print(f"Выжило по прекдикту: {get_sur_amount()}")

    correct = get_correct_predictions(
        "train.csv",
        "submission.csv"
    )

    print(f"Правильных предсказаний: {correct} из {len(data)}")
    print(f"Точность: {correct / len(data) * 100:.2f} %")

if __name__ == "__main__":
    main_func()
