import time
import numpy as np
import torch
from main import ImageBuffer, ImageBufferFast, ImageBufferUltraFast

buffer = ImageBufferUltraFast(100, 3, 4, 4)

times: list[float] = []
for i in range(100):
    start = time.time()
    for i in range(1000):
        buffer(torch.zeros(1, 3, 4, 4, device='cuda'))
    times.append(time.time() - start)
print('Mean: ', np.mean(times))
print('Std: ', np.std(times))
