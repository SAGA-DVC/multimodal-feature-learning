import json

train = json.load(open('../anet_data/legacy/train_data_with_action_classes.json', 'r'))
val_1 = json.load(open('../anet_data/legacy/val_data_1_with_action_classes.json', 'r'))
val_2 = json.load(open('../anet_data/legacy/val_data_2_with_action_classes.json', 'r'))

for key, value in train.items():
    if 'classes' in value.keys():
        classes = value['classes']
        for i, class_num in enumerate(classes):
            if class_num != 0:
                classes[i] = 1

for key, value in val_1.items():
    if 'classes' in value.keys():
        classes = value['classes']
        for i, class_num in enumerate(classes):
            if class_num != 0:
                classes[i] = 1

for key, value in val_2.items():
    if 'classes' in value.keys():
        classes = value['classes']
        for i, class_num in enumerate(classes):
            if class_num != 0:
                classes[i] = 1

json.dump(train, open('../anet_data/train_data_with_binary_classes.json', 'w'))
json.dump(val_1, open('../anet_data/val_1_data_with_binary_classes.json', 'w'))
json.dump(val_2, open('../anet_data/val_2_data_with_binary_classes.json', 'w'))


