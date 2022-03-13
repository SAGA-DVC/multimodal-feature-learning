'''
Implementation taken from https://github.com/YuanGongND/ast/
'''

import torch
import torchaudio

def aframes_to_fbank(aframes: torch.Tensor, sample_frequency, num_mel_bins, target_length, ):
    aframes = aframes - aframes.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        aframes, 
        htk_compat=True,
        sample_frequency=sample_frequency, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=num_mel_bins, 
        dither=0.0, 
        frame_shift=10)
    
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    
    return fbank