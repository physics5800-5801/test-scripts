import piplates.DAQC2plate as DAQC2
import time


DAQC2.setDAC(0,0,2.5)
time.sleep(0.25)
AIN = DAQC2.getADC(0,0)
time.sleep(0.25)
DAQC2.setDAC(0,0,0)

print("AIN =", AIN, "volts")
