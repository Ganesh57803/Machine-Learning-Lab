import pandas as pd
from pprint import pprint
from sklearn.feature_selection import mutual_info_classif
from collections import Counter

def ID3_decision_tree(data_frame, target_attribute, attributes, default_class=None):
    class_counts = Counter(x for x in data_frame[target_attribute])
    
    if len(class_counts) == 1:
        return next(iter(class_counts))
    
    elif data_frame.empty or (not attributes):
        return default_class
    else:
        gains = mutual_info_classif(data_frame[attributes], data_frame[target_attribute], discrete_features=True)
        index_of_max_gain = gains.tolist().index(max(gains))
        best_attribute = attributes[index_of_max_gain]
        decision_tree = {best_attribute: {}}
        remaining_attributes = [i for i in attributes if i != best_attribute]

        for attribute_value, subset_data in data_frame.groupby(best_attribute):
            sub_tree = ID3_decision_tree(subset_data, target_attribute, remaining_attributes, default_class)
            decision_tree[best_attribute][attribute_value] = sub_tree

        return decision_tree

data_frame = pd.read_csv("ID3.csv")

attributes = data_frame.columns.tolist()
print("List of attribute names")
attributes.remove("Target")

for col_name in data_frame.select_dtypes("object"):
    data_frame[col_name], _ = data_frame[col_name].factorize()

print(data_frame)

tree = ID3_decision_tree(data_frame, "Target", attributes)
print("The tree structure")
pprint(tree)
