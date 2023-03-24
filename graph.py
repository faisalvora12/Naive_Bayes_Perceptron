import matplotlib.pyplot as plt 
import numpy as np 
 
import matplotlib.pyplot as plt
import numpy as np
iters = [100, 500, 1000, 5000, 10000, 20000]
train_percent = [54.38, 72.26, 76.64, 77.53, 79.91, 79.78 ]
plt.plot(iters, train_percent)
plt.ylabel('Training set percentages', fontsize=18)
plt.xlabel('Vocabulary Size', fontsize=16)
plt.show()