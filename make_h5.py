import numpy as np
import h5py
import torch

hf = h5py.File('video_features.h5', 'w')

a = torch.rand(3000, 768)

# # sample for train dataset
# hf.create_dataset('v_QOlSCBRmfWY', data=a)
# hf.create_dataset('v_ehGHCYKzyZ8', data=a)
# hf.create_dataset('v_nwznKOuZM7w', data=a)
# hf.create_dataset('v_ogQozSI5V8U', data=a)
# hf.create_dataset('v_nHE7u40plD0', data=a)
# hf.create_dataset('v_69IsHpmRyfk', data=a)

# sample for val dataset
hf.create_dataset('v_uqiMw7tQ1Cc', data=a)
hf.create_dataset('v_bXdq2zI1Ms0', data=a)
hf.create_dataset('v_FsS_NCZEfaI', data=a)
hf.create_dataset('v_K6Tm5xHkJ5c', data=a)
hf.create_dataset('v_4Lu8ECLHvK4', data=a)
hf.create_dataset('v_HWV_ccmZVPA', data=a)

