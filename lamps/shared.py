## Shared variables for multiprocessing
## Warning: Vanilla pyUni10-1.0 does not support multiprocessing
##          Always set USE_MP = False when using vanilla pyUni10
import os

USE_MP = False
PROCS = 4
PHI = []
PHI_RN = []
W = []
TL = []
NET = {}
NETDIR = os.path.dirname(os.path.abspath(__file__)) + "/networks"
