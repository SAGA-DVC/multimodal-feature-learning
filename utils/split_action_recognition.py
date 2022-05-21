import json
import os

base_path = '../anet_data/action_recognition/'

with open('../anet_data/action_recognition/activity_net.v1-3.min.json') as f:
    obj = json.load(f)

database = obj['database']

train, val, test = {}, {}, {}
for key in database.keys():
    to_append = {
        "duration": database[key]['duration'],
        "annotations": database[key]['annotations']
    }

    v_key = f"v_{key}"

    if database[key]["subset"] == 'training':
        train[v_key] = to_append
    elif database[key]["subset"] == 'validation':
        val[v_key] = to_append
    else:
        test[v_key] = to_append

    
with open(os.path.join(base_path, 'train.json'), 'w') as f:
    json.dump(train, f)

with open(os.path.join(base_path, 'val.json'), 'w') as f:
    json.dump(val, f)

with open(os.path.join(base_path, 'test.json'), 'w') as f:
    json.dump(test, f)

