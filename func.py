
def all_status(data: list, column1: str) -> str:
    status_dict = set()
    for obj in data:
        status_dict.add(obj[column1])
    return status_dict

def survived_in_group(data: list, column: str, value: str) -> tuple:
    """Функция возврщает колчиество выжевших, которые соотвествуют признаку value"""
    count_people = 0
    count_survived = 0
    
    
    for obj in data:
        if obj[column] == value:
            count_people +=1 
            if obj['Survived'] == "1":
                count_survived += 1
                
    return (count_people, count_survived)
    