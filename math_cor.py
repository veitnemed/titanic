from storage import get_full_csv_list
from config import TRAIN, CORRELATION_FEATURES
from features import replace_feature_values, get_column_in_matrix, replace_median_ages
data = get_full_csv_list(TRAIN)
data = replace_feature_values(data,"Sex",{"female": 1, "male": 0})
data = replace_median_ages(data)
# >> 891


# >> ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

def filter_valid_pairs(values1: list, values2: list) -> tuple[list, list]:
    res_1 = []
    res_2 = []
    for  v1, v2 in zip(values1, values2):
        if v1 == "" or v2 == "":
            continue
        res_1.append(v1)
        res_2.append(v2)
    return (res_1, res_2)


def mean_value(features_list: list) -> float:
    "Считаем среднее зачение признака feature, нап. Survived,Pclass"
    return sum(list(map(float,features_list)))/len(features_list)

def pearson_coefficient(values1: list, values2: list) -> float:
    sigma_x = standart_diviation(values1)
    sigma_y = standart_diviation(values2)
    cov = covariation(values1,values2)
    return cov/(sigma_x*sigma_y)
    
def covariation(values1: list, values2: list) -> float:
    "Считаем ковариацию столбцов column_1 и column_2"
    x_column_filt, y_column_filt = filter_valid_pairs(values1, values2)
    mean_x = mean_value(x_column_filt); mean_y = mean_value(y_column_filt)
    
    cov = 0
    for x, y in zip(x_column_filt,y_column_filt): # Итерируемся сразу по двум спискам
        cov += (mean_x - float(x))*(mean_y - float(y))
        
    return cov/len(x_column_filt)

def standart_diviation(values: list):
    res = 0.0
    length = len(values)
    mean_x = mean_value(values)
    
    for x in values:
        res += pow(mean_x-float(x),2)
    
    return pow(res/length, 0.5)

def get_cor_coef(column1: str, column2: str, data: list):
    """"""
    values_1 = get_column_in_matrix(data, column1); values_2 = get_column_in_matrix(data, column2)
    filt_values_1, filt_values_2 = filter_valid_pairs(values_1, values_2)
    
    return pearson_coefficient(filt_values_1, filt_values_2), len(filt_values_1)




def cor_marix(data) -> list:

    result = []
    for i in range(len(CORRELATION_FEATURES )):
        row = []
        for j in range(len(CORRELATION_FEATURES)):
            row.append(round(get_cor_coef(CORRELATION_FEATURES[i], CORRELATION_FEATURES[j],data)[0],2))
        result.append(row)
    return result

def print_matrix(data: list):
    print()
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*[" ", *CORRELATION_FEATURES]), "\n")
    for i in range(len(data)):
        row = []
        row.append(CORRELATION_FEATURES[i])

        for j in range(len(data)):
            row.append(data[i][j])
        
        print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*row), "\n")
            
