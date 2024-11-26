# type: ignore
import pywt
import numpy as np


# denoising with discrete wavelet transform
def dwt_denoise(signal: np.ndarray, wavelet: str ="db8") -> np.ndarray:
  coefs = pywt.wavedec(signal, wavelet)
  for i, _ in enumerate(coefs):
    if i not in [0, 1, 7, 8]:
      continue
    else:
      coefs[i] *= 0
  
  signal_denoised = pywt.waverec(coefs, wavelet)

  return signal_denoised