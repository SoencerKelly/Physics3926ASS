import numpy as np

a = np.array([(1,2),(3,4)], dtype="f,f")

for i in range(a.size): np.delete(a[i],1)

print(a)