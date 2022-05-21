import json
import os

base_path = '../anet_data/action_recognition/'

with open('../anet_data/action_recognition/activity_net.v1-3.min.json') as f:
    obj = json.load(f)
    database = obj['database']

action_labels_dict, inverted_action_labels_dict = {}, {}

class_count = 0
for key in database.keys():
    for annotation in database[key]["annotations"]:
        if annotation['label'] not in action_labels_dict:
            action_labels_dict[annotation['label']] = class_count
            inverted_action_labels_dict[class_count] = annotation['label']

            class_count += 1

action_labels_dict["No action"] = class_count
inverted_action_labels_dict[class_count] = "No action"

with open(os.path.join(base_path, 'action_labels_dict.json'), 'w') as f:
    json.dump(action_labels_dict, f)

with open(os.path.join(base_path, 'inverted_action_labels_dict.json'), 'w') as f:
    json.dump(inverted_action_labels_dict, f)