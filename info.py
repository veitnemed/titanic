
from config import  (DATASET_CSV_PATCH, THREASHOLD)
from scores import (
                    survived_counter, 
                    number_of_prediction,
                    create_dict_scores, 
                    create_dict_binary, 
                    scores_to_sigmoids,
                    )
from model import  calculate_mean_loss, create_dict_loss

def show_main_info_for_training(scores: dict, binary: dict, mean_score: float):
    """Выводится основная информация о прогнозе"""
    print(f"Threshold: {round(mean_score,2)}")
    print("Min score:", round(min(list(scores.values())),2))
    print("Max score:", round(max(list(scores.values())),2))
    print(f"Количество выживших по предсказанию: {survived_counter(binary)}")
    
def row_table_v2(binary: dict, 
                     survived: dict, 
                     message: str, 
                     dataset: str):
    """Выввод для бейслайна"""
    length = len(binary)
    n = number_of_prediction(binary, survived)
    accuracy = round((n / length) * 100, 2)
    print("{:<20} {:<20} {:<20} {:<20}".format(*[message, dataset,f"{accuracy} %","---"]))



def row_table(raw_list: list, 
           weights: dict, 
           survived, 
           message: str, 
           dataset: str,
           features_list: list,
           age_values: list):
    """Табличный выводрезльтата"""
    scores = create_dict_scores(raw_list, weights, features_list, age_values)
    binary = create_dict_binary(scores_to_sigmoids(scores), THREASHOLD)
    length = len(binary)
    n = number_of_prediction(binary, survived)
   
    a = (n / length)*100
    maean_loss = calculate_mean_loss(raw_list, survived, weights, features_list, age_values)
    print("{:<20} {:<20} {:<20} {:<20}".format(*[message, dataset,f"{round(a,2)} %",round(maean_loss,3)]))
   
    
def show_weights(weights, new_weights):
    """Вывод весов"""
    print("WEIGHT CHANGES")
    for feature, val in weights.items():
    
        print(f"{feature}: ")
        for k, v in val.items():
            v1 = new_weights[feature][k]
            print(f"{k}: {v} -> {round(v1,2)}")
        print()
            
    
def show_main_info(train,test,seed, seed_train):
    """Выводит шапку"""
    
    print("\n\nINFORMATION FOR TRAINING")
    print(f"Split seed: {seed}")
    print(f"Split seed: {seed_train}")
    print(f"Train: {len(train)} passengers")
    print(f"Validation: {len(test)} passengers\n\n")

def show_top_n_error(train: list, survived: dict, new_weights: dict, n: int, features_list: list, age_values):
    import pandas as pd
    df = pd.read_csv(DATASET_CSV_PATCH)
    df = df.reset_index()
    df = df.drop(labels = ["Name", "Ticket", "Cabin","index"], axis = 1)
    print(f'TOP LOSS (top {n})')
    dict_loss = create_dict_loss(train, survived, new_weights, features_list,age_values)
    
    for idx, item in enumerate(sorted(dict_loss.items(), key = lambda t: -t[1])):
        id, loss = item
        mask = [x == id for x in df["PassengerId"] ]
        print(f"{id} - {round(loss,3)}")
        print(df.loc[mask], '\n')
        if idx == n - 1:
            return 

def print_is_uppdate_weights(old_loss: float, new_loss: float) -> bool:
     if new_loss < old_loss:
            word = "updated"
     else:
            word = "not updated"
     print(f"Checkpoint: {word}\n\n")

def print_result(
                train_context,
                result_context,
                seed: int
                ):

    show_main_info(train_context["train_data"], train_context["valid_data"], seed, seed)
    print("RESULTS")
    print("{:<20} {:<20} {:<20} {:<20}".format(*["Model","Dataset","Accuracy ","Loss"]))
    row_table_v2(binary = train_context["baseline_predictions"],
                        survived = train_context["train_answers"],
                        message = "Baseline",
                        dataset = "Train")
    row_table(raw_list = train_context["train_data"],
               weights = train_context["start_weights"],
               survived = train_context["train_answers"],
               message = "Before training",
               dataset = "Train",
               features_list = train_context["features_list"],
               age_values = train_context["age_values"])
    row_table(raw_list = train_context["train_data"],
               weights = result_context["trained_weights"],
               survived = train_context["train_answers"],
               message = "After training",
               dataset = "Train",
               features_list = train_context["features_list"],
               age_values = train_context["age_values"]
               )
    row_table(raw_list = train_context["valid_data"],
               weights = result_context["trained_weights"],
               survived = train_context["valid_answers"],
               message = "After training",
               dataset = "Validation",
               features_list = train_context["features_list"],
               age_values = train_context["age_values"]
               )
    print(f"\nTraining time: {result_context["time_train"]} sec")
        #show_weights(weights, new_weights)
    print_is_uppdate_weights(train_context["start_loss"], result_context["train_loss"])
        #show_top_n_error(train, survived, new_weights, 20, features_list, age_values)
           
        #new_data = replace_feature_values(train,"Sex",{"female": 1, "male": 0})
        #new_data = replace_median_ages(new_data)
        #dict_loss = create_dict_loss(train, survived, new_weights, features_list)
        #new_data = add_column_from_dataset(new_data, dict_loss, "Loss")
        #print_matrix(cor_marix(new_data))  