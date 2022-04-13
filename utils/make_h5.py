import numpy as np
import h5py
import torch
import json

hf = h5py.File('audio_features.h5', 'a')
annotation = json.load(open('../activity-net/captions/val_1.json', 'r'))

keys = list(annotation.keys())

invalid_videos = json.load(open('../activity-net/captions/invalid_ids.json'))
keys = [k for k in keys if k not in invalid_videos]


for i, key in enumerate(keys):
    print(i)
    hf.create_dataset(key, data=torch.rand(64, 768))


# # sample for train dataset
# hf.create_dataset('v_QOlSCBRmfWY', data=a)
# hf.create_dataset('v_ehGHCYKzyZ8', data=a)
# hf.create_dataset('v_nwznKOuZM7w', data=a)
# hf.create_dataset('v_ogQozSI5V8U', data=a)
# hf.create_dataset('v_nHE7u40plD0', data=a)
# hf.create_dataset('v_69IsHpmRyfk', data=a)

# # sample for val dataset
# hf.create_dataset('v_uqiMw7tQ1Cc', data=a)
# hf.create_dataset('v_bXdq2zI1Ms0', data=a)
# hf.create_dataset('v_FsS_NCZEfaI', data=a)
# hf.create_dataset('v_K6Tm5xHkJ5c', data=a)
# hf.create_dataset('v_4Lu8ECLHvK4', data=a)
# hf.create_dataset('v_HWV_ccmZVPA', data=a)

