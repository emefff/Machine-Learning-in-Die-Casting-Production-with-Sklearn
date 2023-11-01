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
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
def scrap_due_to_low_values(dataframe, limits_list, set_values_list):
    """
    Low values of three process parameters (v_phase2, pressure_packing, temperature_die)
    generate scrap parts. 

    Parameters
    ----------
    dataframe : TYPE pd.DataFrame
        DESCRIPTION. dataframe for all process values
    counters_list : TYPE list of int
        DESCRIPTION. list with three counters, because we do not know the number
                     of affected parts before executing this function.
    limits_list : TYPE list of float
        DESCRIPTION. list with three entries serving as limits to turn parts
                     with lower values into scrap parts
    set_values_list : TYPE list of floats
        DESCRIPTION. list of set process parameters

    Returns
    -------
    df : TYPE pd.DataFrame
        DESCRIPTION. new dataframe containing scrap parts with reduced volume_shot
    counters_list : TYPE list of int
        DESCRIPTION. list of len=3 with counted parts affected by low process 
                     parameters that lead to scrap.

    """
    df = dataframe
    part_ok_list = [] # =1 if part ok, =0 if part is scrap
    
    num_parts_scrap_vphase2 = 0
    num_parts_scrap_pressure_packing = 0
    num_parts_scrap_temperature_die = 0
    
    percentage_v_phase2 = limits_list[0]
    percentage_press_packing = limits_list[1]
    temperature_limit_die = limits_list[2]
    
    velocity_phase2_set = set_values_list[0]
    pressure_packing_set = set_values_list[1]
    
    for i in range(NUMBER_OF_SHOTS):
        # when velocity is low, more defect parts due to insufficient fill
        if df["v_phase2"][i] < percentage_v_phase2 * velocity_phase2_set:
            part_ok = 0
            num_parts_scrap_vphase2 += 1
        else:
            part_ok = 1
        # packing pressure low --> insufficient filling of part
        if df["pressure_packing"][i] < percentage_press_packing * pressure_packing_set:
            part_ok = 0
            num_parts_scrap_pressure_packing += 1
        else:
            pass
        # die temperature < 235°C --> cold flow in some locations on part
        if df["temperature_die"][i] < temperature_limit_die:
            part_ok = 0
            num_parts_scrap_temperature_die += 1
        else:
            pass
        part_ok_list.append(part_ok)
    
    df["part_ok"] = part_ok_list 
    
    counters_list = [0,0,0]
    counters_list[0] = num_parts_scrap_vphase2
    counters_list[1] = num_parts_scrap_pressure_packing
    counters_list[2] = num_parts_scrap_temperature_die
       
    return df, counters_list


def generate_low_press_phase2_values(dataframe, PRESS_PHASE2_FACTOR_LIST, set_values_list):
    """
    Inserts some values into the dataframe with low pressure_phase2. The part number
    is chosen randomly. Target value of these parts is set to zero (scrap part).

    Parameters
    ----------
    dataframe : TYPEpd.DataFrame
        DESCRIPTION. dataframe for all process values
    PRESS_PHASE2_FACTOR_LIST : TYPE list of floats < 1.0
        DESCRIPTION. list of factors will be multiplied with pressure_phase2 
                     values. The purpose is to reduce pressure_phase2              
    set_values_list : TYPE list of floats
        DESCRIPTION. list of set process parameters

    Returns
    -------
    df : TYPE pd.DataFrame
        DESCRIPTION. new dataframe containing scrap parts with reduced pressure_phase2
    parts_changed : TYPE list of int
        DESCRIPTION. unordered list of parts that were changed via this function.

    """
    
    df = dataframe
    # we insert 'low pressure in phase 2" events, maybe due to a blown piston ring at the end
    # of production during training of the model
    num_parts_affected = len(PRESS_PHASE2_FACTOR_LIST)
    count_parts_affected = 0
    
    
    parts_changed = []
        
    num_shots = set_values_list[5]
    # that the model needs a certain amount of scrap parts to this feature correctly
    for i in range(num_shots): # we won't need as many as these....
        random_number = np.random.randint(0, num_shots)
        if part_ok_list[random_number] == 1:
            new_pressure = PRESS_PHASE2_FACTOR_LIST[count_parts_affected] * df["pressure_phase2"][random_number]
            df["pressure_phase2"][random_number] = new_pressure
            part_ok_list[random_number] = 0
            count_parts_affected += 1
            parts_changed.append(random_number)
        else:
            pass
        if count_parts_affected == num_parts_affected:
            break
    
    df["part_ok"] = part_ok_list
    return df, parts_changed


def generate_low_volume_values(dataframe, VOLUME_FACTOR_LIST, set_values_list):
    """
    Inserts some values into the dataframe with low volume_shot. The part number
    is chosen randomly. Target value of these parts is set to zero (scrap part).

    Parameters
    ----------
    dataframe : TYPE pd.DataFrame
        DESCRIPTION. dataframe for all process values
    VOLUME_FACTOR_LIST : TYPE list of floats < 1.0
        DESCRIPTION. list of factors will be multiplied with volume_shot 
                     values. The purpose is to reduce the volume_shot                      
    set_values_list : TYPE list of floats
        DESCRIPTION. list of set process parameters

    Returns
    -------
    df : TYPE pd.DataFrame
        DESCRIPTION. new dataframe containing scrap parts with reduced volume_shot
    parts_changed : TYPE list of int
        DESCRIPTION. unordered list of parts that were changed via this function.

    """
    df = dataframe
    num_parts_affected = len(VOLUME_FACTOR_LIST)
    count_parts_affected = 0
    
    parts_changed = []
        
    num_shots = set_values_list[5]
    for i in range(num_shots): # we won't need as many as these....
        random_number = np.random.randint(0, num_shots)
        if part_ok_list[random_number] == 1:
            new_volume = VOLUME_FACTOR_LIST[count_parts_affected] * df["volume_shot"][random_number]
            df["volume_shot"][random_number] = new_volume
            part_ok_list[random_number] = 0
            count_parts_affected += 1
            parts_changed.append(random_number)
        else:
            pass
        if count_parts_affected == num_parts_affected:
            break
        
    df["part_ok"] = part_ok_list
    return df, parts_changed

###############################################################################        
###############################################################################

VELOCITY_PHASE2_SET = 4000 # mm/s
VELOCITY_PHASE2_DEVIATION = 100 # mm/s, standard deviation of VELOCITY_PHASE2_SET

POS_SHOT_START_SET = 65 # mm
POS_SHOT_START_DEVIATION = 2 # mm, standard deviation of POS_SHOT_START_SET

POS_SHOT_END_SET = 165 # mm
POS_SHOT_END_DEVIATION = 2 # mm,  standard deviation of POS_SHOT_END_SET

PRESSURE_PACKING_SET = 425 # bar
PRESSURE_PACKING_DEVIATION = 25 # bar, standard deviation of PRESSURE_PACKING_SET 

PRESSURE_PHASE2 = 50 # bars; this value is determined by gate size and flow conditions in the tool, but we set it here.
PRESSURE_PHASE2_DEVIATION = 2 # bars, standard deviation of PRESSURE_PHASE2

TEMPERATURE_DIE_SET = 260 # °C
TEMPERATURE_DIE_DEVIATION = 10 # °C, standard deviation of TEMPERATURE_DIE_SET
# this is very simplified here, the die temperature has a very complex distribution
# and a time lag of minutes or even hours from setting it to reaching the set temp.
# in the die.

DIAMETER_PISTON = 90 # mm

VOLUME_SHOT_SET = (POS_SHOT_END_SET-POS_SHOT_START_SET)*(DIAMETER_PISTON/2)**2 * np.pi
######################### GENERATION OF SHOT DATA #############################
# in real-life this would be data from the machine
# it is used to train the model
# we take the production lot of one day (3 shifts, 1 planned hour of maintenance per shift)

cycle_time = 30 # s, time it takes for a single part
parts_per_min = 60 / cycle_time
num_parts_per_day = int(7 * 3 * 60 * parts_per_min)
print(f"{num_parts_per_day = }\n")

NUMBER_OF_SHOTS = num_parts_per_day * 2 # data of two production days

v_phase2_list = []
pos_shot_start_list = []
pos_shot_end_list = []
pressure_packing_list = []
pressure_phase2_list = []
temperature_die_list = []
volume_shot_list = []


v_phase2_list = np.random.normal(loc=VELOCITY_PHASE2_SET, scale=VELOCITY_PHASE2_DEVIATION, size=NUMBER_OF_SHOTS) # mm/s
pos_shot_start_list = np.random.normal(loc=POS_SHOT_START_SET, scale=POS_SHOT_START_DEVIATION, size=NUMBER_OF_SHOTS) # mm
pos_shot_end_list = np.random.normal(loc=POS_SHOT_END_SET, scale=POS_SHOT_END_DEVIATION, size=NUMBER_OF_SHOTS) # mm
pressure_packing_list = np.random.normal(loc=PRESSURE_PACKING_SET, scale=PRESSURE_PACKING_DEVIATION, size=NUMBER_OF_SHOTS) # bar
pressure_phase2_list = np.random.normal(loc=PRESSURE_PHASE2, scale=PRESSURE_PHASE2_DEVIATION, size=NUMBER_OF_SHOTS) # °bar
temperature_die_list = np.random.normal(loc=TEMPERATURE_DIE_SET, scale=TEMPERATURE_DIE_DEVIATION, size=NUMBER_OF_SHOTS) # °C
volume_shot_list = (pos_shot_end_list-pos_shot_start_list)*(DIAMETER_PISTON/2)**2*np.pi # mm³
#print(f"{len(volume_shot_list) = }")
#print("Set volume is = ", (POS_SHOT_END_SET-POS_SHOT_START_SET)*(DIAMETER_PISTON/2)**2*np.pi, "\n")

# we create a dataframe from the lists above
# the two positions are not necessary, the are not independent variables as
# they are connected to the shot volume
df = pd.DataFrame({'v_phase2' : v_phase2_list})
# df["pos_shot_start"] = pos_shot_start_list
# df["pos_shot_end"] = pos_shot_end_list
df["pressure_packing"] = pressure_packing_list
df["pressure_phase2"] = pressure_phase2_list
df["temperature_die"] = temperature_die_list
df["volume_shot"] = volume_shot_list

print(df.head())
print(df.info())
print("")


########################## GENERATING TARGET DATA #############################
# we need a target for ML, we will generate some data of good and bad parts artificially
# in real-life, these data would of course have to be found by quality control
# by measurement of the part or visual or X-ray inspection
# therefore the traget can have one of two values: 1 = good part, 0 = scrap part

# some counters that are handy
num_parts_scrap_vphase2 = 0
num_parts_scrap_pressure_packing = 0
num_parts_scrap_temperature_die = 0
part_ok_list = [] # =1 if part ok, =0 if part is scrap

percentage_limit_v_phase2 = 0.95
percentage_limit_press_packing = 0.875
temperature_limit_die = 235

count_list = [num_parts_scrap_vphase2, num_parts_scrap_pressure_packing, num_parts_scrap_temperature_die]
limits_list = [percentage_limit_v_phase2, percentage_limit_press_packing, temperature_limit_die]
set_values_list = [VELOCITY_PHASE2_SET, PRESSURE_PACKING_SET, PRESSURE_PHASE2, \
                   TEMPERATURE_DIE_SET, DIAMETER_PISTON, NUMBER_OF_SHOTS]


# generate some scrap values due to low process values
df, count_list = scrap_due_to_low_values(df, limits_list, set_values_list)
part_ok_list = df["part_ok"]


parts_changed_press_phase2 = []
PRESS_PHASE2_FACTOR_LIST = [np.random.uniform(0.8, 0.99) for i in range(50)] # generate some values < 1 for short shots
df, parts_changed_press_phase2 = generate_low_press_phase2_values(df, PRESS_PHASE2_FACTOR_LIST, set_values_list)
part_ok_list = df["part_ok"]


parts_changed_volume = []
VOLUME_FACTOR_LIST = [np.random.uniform(0.8, 0.99) for i in range(50)] # generate some values < 1 for short shots
df, parts_changed_volume = generate_low_volume_values(df, VOLUME_FACTOR_LIST, set_values_list)
part_ok_list = df["part_ok"]

# print some scrap numbers
num_parts_scrap_vphase2 = count_list[0]
num_parts_scrap_pressure_packing = count_list[1]
num_parts_scrap_temperature_die = count_list[2]
NUM_PARTS_SCRAP_LOW_PRESS_PHASE2 = len(PRESS_PHASE2_FACTOR_LIST)
NUM_PARTS_SCRAP_VOLUME_SHOT = len(VOLUME_FACTOR_LIST)
print(f"{num_parts_scrap_vphase2 = }")
print(f"{num_parts_scrap_pressure_packing = }")
print(f"{num_parts_scrap_temperature_die = }")
print(f"{NUM_PARTS_SCRAP_LOW_PRESS_PHASE2 = }")
print(f"{NUM_PARTS_SCRAP_VOLUME_SHOT = }")

# how much scrap do we get from the artificial data?
number_of_scrap_parts = df["part_ok"].value_counts()[0]
number_of_ok_parts = df["part_ok"].value_counts()[1]
percentage_scrap = number_of_scrap_parts/number_of_ok_parts * 100
print(f"{percentage_scrap = } %\n")

# some plots to check value distributions
BINS = 25
plots = 'yes' # yes or no
plots = 'no'

if plots == 'yes': # A LOT OF PLOTS!
    plt.figure(figsize=(15,9))
    plt.hist(v_phase2_list, bins=BINS, label='v_phase2')
    plt.legend()
    plt.xlabel('v_phase2 / mm/s')
    plt.ylabel('Number of shots')
    plt.show()
    
    plt.figure(figsize=(15,9))
    plt.hist(pos_shot_start_list, bins=BINS, label='pos_shot_start')
    plt.legend()
    plt.xlabel('pos_shot_start / mm')
    plt.ylabel('Number of shots')
    plt.show()
    
    plt.figure(figsize=(15,9))
    plt.hist(pos_shot_end_list, bins=BINS, label='pos_shot_end')
    plt.legend()
    plt.xlabel('pos_shot_end / mm')
    plt.ylabel('Number of shots')
    plt.show()
    
    plt.figure(figsize=(15,9))
    plt.hist(pressure_packing_list, bins=BINS, label='pressure_packing')
    plt.legend()
    plt.xlabel('pressure_packing / bar')
    plt.ylabel('Number of shots')
    plt.show()
    
    plt.figure(figsize=(15,9))
    plt.hist(pressure_phase2_list, bins=BINS, label='pressure_phase2')
    plt.legend()
    plt.xlabel('pressure_phase2 / bar')
    plt.ylabel('Number of shots')
    plt.show()
    
    # look at distribution of ok parts
    plt.figure(figsize=(15,9))
    plt.hist(part_ok_list, bins=BINS, label='part_ok')
    plt.legend()
    plt.xlabel('part_ok')
    plt.ylabel('Number of shots')
    plt.show()
    
else:
    pass

# print(df)
# print(f"{df.info() = }\n")    

##################### MACHINE LEARNING WITH RANDOM FOREST #####################
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

# let's do a heatmap of the df correlations
plt.figure(figsize = (15,9))
sns.heatmap(df.corr(), cmap = "Spectral", annot = True)
plt.show()

# Splitting dataframe into independent and target values
X = df.drop(["part_ok"], axis = 1) # drop the target value for the independent data
y = df.part_ok # only the target value in y

# split into testing and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# declaring the model
rf_model = RandomForestClassifier()

# printing some info about the model
#print(rf_model.__dir__())
print(f"{rf_model.n_estimators = }")
#print(f"{rf_model.estimator_params = }")
print("")

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

print(f"{accuracy_score(y_test, y_pred) = }")
print("")
cv_scores = cross_val_score(estimator = rf_model, X = X_train, y = y_train, cv = 10)
print(f"{cv_scores.mean() = }")
print("")

print(f"{confusion_matrix(y_test, y_pred) = }")
print("")
# rows are actual labels
# columns are predicted labels

# better graph this....
plt.figure(figsize = (15,9))
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cnf_matrix, annot = True, cmap = "YlGnBu") # currently, there's a bug in seaborn that doesn't print all annots
# https://github.com/microsoft/vscode-jupyter/issues/14363
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))

# due to our artificial data generated for part_ok (lines 84 and following lines)
# the model is already very good. A random forest is not fooled by such simple 
# conditions. Therefore, the model does not need further optimization. In real-life
# data, this is for sure necessary.
    
############ SHOW SOME DECISION TREES OF THE RANDOM FOREST MODEL ##############
# from sklearn import tree
# max_depth = 5
# number_of_trees = 2
# for i in range(number_of_trees):
#     plt.figure(figsize=(15, 9))
#     tree.plot_tree(rf_model.estimators_[i], filled = True, \
#                     feature_names=X_train.columns, max_depth=max_depth)

###################### SAVE THE MODEL IN CURRENT WORKING DIR ##################
from joblib import dump
dump(rf_model, 'Die_Casting_RF_model.joblib') 
