import numpy as np
import matplotlib.pylab as plt
t1 = [1,2,3,4]
t2 = [1,2,3,4]
t3 = [1,2,5,4]
t4 = [1,2,3,4]

plt.figure(12)
plt.plot(t1, t2)
plt.ylabel('Feature Importance Score')
plt.show()


plt.plot(t3,t4)
plt.ylabel('Feature ')
plt.show()

plt.figure(33)
plt.subplot(2_1)
plt.plot(t1, t2)

plt.subplot(222)
plt.plot(t3, t4)
plt.show()



