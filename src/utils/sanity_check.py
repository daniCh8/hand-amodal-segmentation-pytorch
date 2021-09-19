import torch
print('PyTorch version:')
print(torch.__version__)
print('CUDA version:')
print(torch.version.cuda)
print('Num GPU detected:')
print(torch.cuda.device_count())
