d = {"Sex": { "female": 0.742,"male": 0.189
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
    }}


def func(a):
    res = 0
    for val in a.values():
        for v in val.values():
            res += v
    return res

target = func(create_new_weights(d,0.1))
error = 0
iter = 0
k = 0
for i in range(1000):
    k = i
    now_sum = func(create_new_weights(d,0.1))
    if abs(now_sum - target) > 0.0001:
        iter += 1
        error += abs(now_sum - target)/iter
        break

print(f"i: {k}")
if k == 999:
    print(True)
if iter != 0:
    print(f"Error: {error/iter}")

target_w = create_new_weights(d,0.1)
mutations = 0
for _ in range(1000):

    new_w = create_new_weights(d,0.1)
    if d != new_w:
        mutations += 1

print (f"Mutations: {mutations}")

def best_threshold(raw_list: list, 
                   actual_survived: dict, 
                   steps: int, 
                   train_weights: dict, 
                   start_threashold) :
  
    best_threshold = start_threashold
    beast_evaluate = 0
    range_values = float_range(start_threashold - start_threashold/2, start_threashold + start_threashold/2, steps,3)
    for value in range_values:
        evaluate = evaluate_weights(raw_list, actual_survived, train_weights, value)
        if evaluate > beast_evaluate:
            beast_evaluate = evaluate
            best_threshold = value
    return best_threshold