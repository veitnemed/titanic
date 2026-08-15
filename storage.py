import csv


def get_csv_dict(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    return data

def save_csv_predict(d: dict):
    """Сохранение csv предикта """
    columns = ["PassengerId", "Score"]

    with open("result.csv", "w", encoding="utf-8", newline="") as f:
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