from storage import survived_dict, get_train_csv_lists
from features import replace_median_ages
from config import  (TRAIN, DEFAULT_WEIGHTS,STEPS_FOR_TRAIN,NUMBER_OF_ITERATIONS,
                     SEED_SPLIT,
                     SEED_TRAIN)
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
        train, test  = get_train_csv_lists(TRAIN, seed_value = SEED_SPLIT)
        train, test = replace_median_ages(train), replace_median_ages(test)
        survived = survived_dict(train)
        
        new_weights, _ = train_classifier(raw_list = train,
                                              actual = survived,
                                              weights = default_weights,
                                              steps = STEPS_FOR_TRAIN,
                                              iters = NUMBER_OF_ITERATIONS,
                                              seed_value = SEED_TRAIN,
                                              features_list = features_list,
                                              age_values = age_values) 
        result.append(new_weights)
        
    return  result[0] == result[1]

def main_tests():
    assert reproducibility() is True

if __name__ == "__main__":
    main_tests()