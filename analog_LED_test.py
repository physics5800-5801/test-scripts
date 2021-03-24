import piplates.DAQC2plate as DAQC2
import numpy as np
import time


voltages = np.arange(2, 4, 0.1)

for v in voltages:
    DAQC2.setDAC(0,0,v)
    time.sleep(0.25)
    DAQC2.setDAC(0,0,0)
    time.sleep(0.25)
    print("Vout = ", v)
