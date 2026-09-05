from storage import (
                     survived_dict,
                     baseline_dict, 
                     get_train_csv_lists,
                     save_json,
                     load_json, 
                     create_json, json_init, clean_json)

from features import replace_median_ages
from config import  (TRAIN, 
                     DEFAULT_WEIGHTS,
                     STEPS_FOR_TRAIN,
                     NUMBER_OF_ITERATIONS,
                     WEIGHTS,
                     SEED_SPLIT,
                     SEED_TRAIN)

from info import (show_main_info,
                  row_table,
                  row_table_v2,
                  print_is_uppdate_weights,
                  show_top_n_error,
                  show_weights
                  )

from model import train_classifier, calculate_mean_loss
import os

def main_train(seed):
    """Главная функция преокта"""
    
    default_weights = DEFAULT_WEIGHTS.copy()
    age_values = None
    if "Age" in default_weights:
        age_values = list(map(int,list(default_weights["Age"].keys())))
    # 1) Начальные рассчёты
    json_init(WEIGHTS, DEFAULT_WEIGHTS)

    features_list = list(default_weights.keys())
    weights = load_json(WEIGHTS)
    train, test  = get_train_csv_lists(TRAIN, seed_value = seed)
    train, test = replace_median_ages(train), replace_median_ages(test)
    survived = survived_dict(train)
    baseline = baseline_dict(train)
    survived_test = survived_dict(test)
    start_loss = calculate_mean_loss(train, survived, weights, features_list, age_values)
    
    # 2) Вывод шапки
    show_main_info(train, test, seed, seed)
    
    # 3) Обучение
    print("TRAINING")
    new_weights, time_train = train_classifier(raw_list = train,
                                      actual = survived,
                                      weights = weights,
                                      steps = STEPS_FOR_TRAIN,
                                      iters = NUMBER_OF_ITERATIONS,
                                      seed_value = seed,
                                      features_list = features_list,
                                      age_values = age_values,
                                      show_progress = True)
    
    # 4) Результаты 
    print("\n\nRESULTS")
    print("{:<20} {:<20} {:<20} {:<20}".format(*["Model","Dataset","Accuracy ","Loss"]))
    row_table_v2(binary = baseline,
                    survived = survived,
                    message = "Baseline",
                    dataset = "Train")
  
    row_table(raw_list = train,
           weights = weights,
           survived = survived,
           message = "Before training",
           dataset = "Train",
           features_list=features_list,
           age_values = age_values)
    
    row_table(raw_list = train,
           weights = new_weights,
           survived = survived,
           message = "After training",
           dataset = "Train",
           features_list = features_list,
           age_values=age_values
           )
    row_table(raw_list = test,
           weights = new_weights,
           survived = survived_test,
           message = "After training",
           dataset="Validation",
           features_list = features_list,
           age_values = age_values
           )
    save_json(WEIGHTS, new_weights)
    
    # 5) Время рассчёта и апдейт
    print(f"\nTraining time: {time_train} sec")
    new_loss = calculate_mean_loss(test,survived_test,new_weights,features_list, age_values)
    
    #show_weights(weights, new_weights)
    print_is_uppdate_weights(start_loss, new_loss)
    #show_top_n_error(train, survived, new_weights, 20, features_list, age_values)
   
    #new_data = replace_feature_values(train,"Sex",{"female": 1, "male": 0})
    #new_data = replace_median_ages(new_data)
    #dict_loss = create_dict_loss(train, survived, new_weights, features_list)
    #new_data = add_column_from_dataset(new_data, dict_loss, "Loss")
    #print_matrix(cor_marix(new_data))  
    return new_loss

def impact_weghts(start_loss, seed):
    
    print("ВКЛАДЫ ВЕСОВ\n")
    
    print(f"Изначальный LOSS: {round(start_loss, 3)}")
    print(f"Seed: {seed}")
    
    default_weights = DEFAULT_WEIGHTS.copy()
    res = {}
    for k, v in DEFAULT_WEIGHTS.items():
        clean_json(WEIGHTS)
        default_weights.pop(k)
    
        if "Age" in default_weights:
            age_values = list(map(int,list(default_weights["Age"].keys())))
            
        # 1) Начальные рассчёты
        json_init(WEIGHTS, DEFAULT_WEIGHTS)
    
        features_list = list(default_weights.keys())
        weights = load_json(WEIGHTS)
        train, test  = get_train_csv_lists(TRAIN, seed_value = seed)
        train, test = replace_median_ages(train), replace_median_ages(test)
        survived = survived_dict(train)
        survived_test = survived_dict(test)

        new_weights, time_train = train_classifier(raw_list = train,
                                          actual = survived,
                                          weights = weights,
                                          steps = STEPS_FOR_TRAIN,
                                          iters = NUMBER_OF_ITERATIONS,
                                          seed_value = seed,
                                          features_list = features_list,
                                          age_values = age_values)  
        
        new_loss = calculate_mean_loss(test, survived_test, new_weights, features_list, age_values)
        res[k] = round(new_loss,3)
        default_weights[k] = v
    
    for k, v in res.items():
        print(f"Loss без {k}: {v} // Польза {100*round(v-start_loss,4)} % ")
        
        
    
    
   
    
    
      
def main_func():
    #for idx, seed in enumerate(list(range(10))):
        #print(f"Эксперимент {idx+1}")
        #start_loss = main_train(seed)
        #impact_weghts(start_loss,seed)
    main_train(10)
if __name__ == "__main__":
    os.system("cls")
    main_func()
    

