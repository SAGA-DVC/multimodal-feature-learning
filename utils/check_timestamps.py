import json

a = json.load(open('anet_data/train.json', 'r'))

for key in a.keys():
    for t in a[key]['timestamps']:
        if t[0] >= t[1] :
            print(key)

# v_rhOtqArO-3Y
# v_N7ppHQNikv8
# v_4rKTw99bM8g
# v_0bosp4-pyTM