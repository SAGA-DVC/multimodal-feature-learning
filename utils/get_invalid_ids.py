import json
import os

invalid_ids = []

train = '../activity-net/captions/train.json'
val_1 = '../activity-net/captions/val_1.json'
val_2 = '../activity-net/captions/val_2.json'

a = json.load(open(train))

for i, key in enumerate(a.keys()):
    print(i)
    if not os.path.exists(f'../activity-net/30fps_splits/train/{key}.mp4') and not os.path.exists(f'../activity-net/30fps_splits/train/{key}.mkv'):
        invalid_ids.append(key)


b = json.load(open(val_1))

for i, key in enumerate(b.keys()):
    print(i)
    if not os.path.exists(f'../activity-net/30fps_splits/val/{key}.mp4') and not os.path.exists(f'../activity-net/30fps_splits/val/{key}.mkv'):
        invalid_ids.append(key)


c = json.load(open(val_2))

for i, key in enumerate(c.keys()):
    print(i)
    if not os.path.exists(f'../activity-net/30fps_splits/val/{key}.mp4') and not os.path.exists(f'../activity-net/30fps_splits/val/{key}.mkv'):
        invalid_ids.append(key)

json.dump(invalid_ids, open('../activity-net/captions/invalid_ids.json', 'w'))


