import numpy as np
import matplotlib.pyplot as plt
import scipy

Xup = np.arange(0,360,1)
Xdown = np.arange(360,0,-1)

np.corrcoef(Xup, Xdown)[0,1]

rs = []
for _ in range(10000):
    rs.append(np.corrcoef(Xup, np.roll(Xdown, np.random.randint(1,360, size=1)))[0,1])

print(np.mean(rs))
print(scipy.stats.ttest_1samp(rs, popmean=0))
plt.hist(rs, bins=100)
plt.show()
