import os
import sys
import numpy as np

# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from model.models import LoadModel 

load = LoadModel()

mdl = load.load_model()

x_test = [6.93057953e+01, 3.26680000e+04, 6.00000000e+00, 1.95700000e+03,
       1.97500000e+03, 2.51500000e+03, 3.00000000e+00, 0.00000000e+00,
       4.00000000e+00, 9.00000000e+00]

print(np.expm1(mdl.predict(np.array(x_test).reshape(1, 10))))
