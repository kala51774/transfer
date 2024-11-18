import torch
import numpy as np
import pywt
import ptwt  # use "from src import ptwt" for a cloned the repo

# generate an input of even length.
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
data_torch = torch.from_numpy(data.astype(np.float32))
# forward
matrix_wavedec = ptwt.MatrixWavedec(pywt.Wavelet("haar"), level=2)
coeff = matrix_wavedec(data_torch)
print(coeff)
# backward
matrix_waverec = ptwt.MatrixWaverec(pywt.Wavelet("haar"))
rec = matrix_waverec(coeff)
print(rec)