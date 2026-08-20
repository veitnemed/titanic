
def repalace_feature_values(data: list, feature, replace_dict: dict):
    for p in data:
        p[feature] = replace_dict[p[feature]]
    return data