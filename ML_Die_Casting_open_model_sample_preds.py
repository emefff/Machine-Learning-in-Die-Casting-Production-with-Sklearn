#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:25:26 2023

The example below will show the efficiency of a random forest machine learning
model on die casting data. Only a few parameters are chosen, the model can
be expanded to many more.

The aim is to get a trained model, that can predict if a part is good or scrap.
In real-life, the trained model can be sampled in real-time with machine data.
That means, the running die-casting machine will measure the desired values, 
send them to the model, the model will estimate if the part is good or scrap. 
There is enough time during part cooling and handling for calculation of the result
and if the result is part_ok = 0, sending the part to an extra handling area or
scrap area.

@author: mario
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def make_prediction(part_data):
    """
    Makes a predicition on part_data with rf_model

    Parameters
    ----------
    part_data : TYPE np.array of data
        DESCRIPTION. Part data as np.array.reshape(1, -1)

    Returns
    -------
    None.

    """
    y_pred = rf_model.predict(part_data)
    print(50*"*")
    print("Testing a sample...........")
    print(f"{part_data = }\n")
    if y_pred == [1]:
        print("This part is OK.")
    else:
        print("This part is SCRAP.")
    print(50*"*","\n")    
    return None




###################### LOAD THE MODEL FROM CURRENT WORKING DIR ##################
from joblib import load
rf_model = load('Die_Casting_RF_model.joblib') 

# printing some info about the model
print("Some info about the model.........")
print(50*"-")
#print(f"{rf_model.__dir__() = }")
print(f"{rf_model.n_estimators = }\n")
print(f"{rf_model.feature_names_in_ = }\n")
#print(f"{rf_model.estimators_ = }\n")
print(f"{rf_model.classes_ = }\n")
print(50*"-","\n")

################# PREDICT IF A PART WILL BE SCRAP OR GOOD #####################
# we now just input a single sample part into our model
# inm real-life  this could be data from the running machine
# the model will tell from the measured data if a part is ok or not,
# there is nearly no need for QC if the model is very good!
# Inference is very fast even on a small industrial computer.
# the values must be entered exactly as the model expects

# first we test a good part
make_prediction(np.array([4000, 65, 165, 425, 50, 250, 636172.5]).reshape(1, -1))

# now we test a scrap part (velocity too low...)
make_prediction(np.array([3750, 65, 165, 425, 50, 250, 636172.5]).reshape(1, -1))

# now we test a scrap part (velocity ok, but packing_pressure very low...)
make_prediction(np.array([3850, 65, 165, 300, 50, 250, 636172.5]).reshape(1, -1))

# now we test a scrap part (velocity ok, packing_pressure ok, die temp very low...)
make_prediction(np.array([3850, 65, 165, 425, 50, 220, 636172.5]).reshape(1, -1))
    
# now we test a scrap part (pressure phase 2 very low)
make_prediction(np.array([3850, 65, 165, 425, 10, 255, 636172.5]).reshape(1, -1))