import json
import os

base_path = '../anet_data/action_recognition/'

with open('../anet_data/action_recognition/activity_net.v1-3.min.json') as f:
    obj = json.load(f)

with open('../anet_data/action_recognition/invalid_ids.json') as f:
    invalid_ids = json.load(f)

database = obj['database']

val = {'database' : {}}
count_val_invalid, count_val = 0, 0
for key in database.keys():
    to_append = {
        "duration": database[key]['duration'],
        "annotations": database[key]['annotations']
    }

    v_key = f"v_{key}"

    if database[key]["subset"] == 'validation':
        count_val += 1
        if f'v_{key}' in invalid_ids:
            count_val_invalid += 1
            continue
        val['database'][v_key] = database[key]
    
print(count_val_invalid, count_val)

with open(os.path.join(base_path, 'no_invalid_val.json'), 'w') as f:
    json.dump(val, f)

