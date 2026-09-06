from storage import survived_dict, get_train_csv_lists
from features import replace_median_ages
from config import  (DATASET_CSV_PATH, DEFAULT_WEIGHTS,WEIGHT_STEPS,NO_CHANGE_ITERATIONS,
                     SPLIT_SEED,
                     TRAIN_SEED)
from model import train_classifier

def reproducibility():
    """Проврка на воспроизводимость рандом"""
    default_weights = DEFAULT_WEIGHTS.copy()
    result = []
    for _ in range(2):
    
        age_values = None
        if "Age" in default_weights:
            age_values = list(map(int,list(default_weights["Age"].keys())))
        
    
        features_list = list(default_weights.keys())
        train, test  = get_train_csv_lists(DATASET_CSV_PATH, seed_value = SPLIT_SEED)
        train, test = replace_median_ages(train), replace_median_ages(test)
        survived = survived_dict(train)
        
        new_weights, _ = train_classifier(raw_list = train,
                                              train_answers = survived,
                                              start_weights = default_weights,
                                              steps = WEIGHT_STEPS,
                                              iters = NO_CHANGE_ITERATIONS,
                                              seed_value = TRAIN_SEED,
                                              features_list = features_list,
                                              age_values = age_values) 
        result.append(new_weights)
        
    return  result[0] == result[1]

def main_tests():

    repr =  reproducibility() 
    flags_list = [repr]
    for idx,fg in enumerate(flags_list):
        print(idx+1,") Воспроизводимость рандома:", fg)
        assert fg is True
    
if __name__ == "__main__":
    main_tests()