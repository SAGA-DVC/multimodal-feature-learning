import json
import os

base_path = '../anet_data/action_recognition/'

with open('../anet_data/action_recognition/activity_net.v1-3.min.json') as f:
    obj = json.load(f)
    database = obj['database']

train_class, val_class = {}, {}

class_count = 0
for key in database.keys():
    if database[key]["subset"] == 'training':
        for annotation in database[key]["annotations"]:
            if annotation['label'] not in train_class:
                train_class[annotation['label']] = 1
            else:
                train_class[annotation['label']] += 1
        
    elif database[key]["subset"] == 'validation':
        for annotation in database[key]["annotations"]:
            if annotation['label'] not in val_class:
                val_class[annotation['label']] = 1
            else:
                val_class[annotation['label']] += 1

with open(os.path.join(base_path, 'train_class_count.json'), 'w') as f:
    json.dump(train_class, f, indent=4, sort_keys=True)

with open(os.path.join(base_path, 'val_class_count.json'), 'w') as f:
    json.dump(val_class, f, indent=4, sort_keys=True)