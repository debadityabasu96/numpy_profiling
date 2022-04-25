import time
import numpy as np


np.show_config()
np.random.seed(0)
n = 2000
sum_time = 0

A = np.random.randn(n, n).astype('float64')
B = np.random.randn(n, n).astype('float64')

for i in range(20):
    start_time = time.time()
    nrm = np.linalg.norm(A @ B)
    end_time = time.time()
    sum_time = sum_time + end_time - start_time
    print("took {} seconds ".format(time.time() - start_time))
    print("norm = ",nrm)

average_time = sum_time/20
print("Average Time = " + str(average_time))    