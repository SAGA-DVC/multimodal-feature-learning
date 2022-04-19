import json
import os

invalid_ids = []

# train = '../activity-net/captions/train.json'
# val_1 = '../activity-net/captions/val_1.json'
# val_2 = '../activity-net/captions/val_2.json'


train = 'anet_data/train.json'
val_1 = 'anet_data/val_1.json'
val_2 = 'anet_data/val_2.json'


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

# audio errors
audio_errors = ["v_b1RAYvxWawA", "v_G4mX4StOvQE", "v_YcDlkZkPb6g", "v_7rf06_5zNJk", "v_z9l32VOM6wY", "v_7Iy7Cjv2SAE", "v_0gf3AgK1YLY", "v_5asz3rt3QyQ", "v_QDjaaUtepHo", "v_0BXBfSWIR2k", "v_D2JvqkKa-qM", "v_fmtW5lcdT_0", "v_DXu_aHrZaUs", "v_vvdmMyyAtN0", "v_JQf_oSGY8q4", "v_jNGa0jPAMjI", "v_BSl22Hx2WGM", "v_mFWRIp164r4", "v_4BofYu8Soz8", "v_Mzojo2EeWu8", "v_nHafujMomWg", "v_QjaEDlh805g", "v_QJfuxpFMn8s", "v_u1upxlAgsqM", "v_tD30qafrkhM", "v_PG0ao4HkF8M", "v_8wqlhbw4e30", "v_QXN6odBnVmI", "v_DwaoxjXwC1M", "v_JTGS1YulUQw", "v_jto8_gMKUjE", "v_1OJa2iiFxfk", "v_1PTNnaEu8xo", "v_1v5HE_Nm99g", "v_1T66cuSjizE"]
invalid_ids += audio_errors

json.dump(invalid_ids, open('../activity-net/captions/invalid_ids.json', 'w'))
json.dump(invalid_ids, open('anet_data/invalid_ids.json', 'w'))

print('length', len(invalid_ids))
print(len(a.keys()), len(b.keys()), len(c.keys()))




