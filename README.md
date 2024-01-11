# Machine-Learning-in-Die-Casting-Production-with-Sklearn

This is a simple example on how machine learning could be implemented in a production environment (here: die casting, but it could really be any kind of production). 
There are three files in the folder:

ML_Die_Casting_train_test_save_model.py
    Generates a random forest model of artificially generated die casting data and saves this model in you working dir.
    In a real production environment, the data train on will com from the die casting machine. As always in ML, the better
    your data is, the better your model will be. What does that mean regarding 'real data'? You will have to provoke scrap
    for training the model. Scrap costs money and time.

ML_Die_Casting_open_model_sample_preds.py
    Opens the saved model and makes predictions with machine data entered by the user.

Die_Casting_RF_model.joblib
    An example of a trained model that can be opened with ML_Die_Casting_open_model_sample_preds.py

    
Let's take a brief look at the generated data. For example, take the shot speed in phase 2 (this would equate to an average value in real life data, because the shot speed is almost never a horizontal line):

![Bildschirmfoto vom 2023-11-01 17-05-26](https://github.com/emefff/Machine-Learing-in-Die-Casting-Production-with-Sklearn/assets/89903493/80ec501a-552a-4da5-b043-692d869f3c1d)

As we generate Gaussian values around 4000mm/s (a reasonable value) we get a EXP(-X**2) like distribution of data. Real-life data may have different distributions. Here, we only use the data for training the model.
We also need to create target data artificially, that is, we have to tell the model which values of v_phase2 (or any of the other independent parameters) do not result in a good part. At a real machine, reasons for v_phase2 being insufficient can be anything related to the hydraulics of the machine. For example: a valve could be damaged, leading to a very low shot speed v_phase2.
Here we artificilly tell the model: any value of v_phase2 below 95% of the set value (4000mm/s) will lead to a scrap art. The target data 'part_ok' then is equal to 0. Good parts are assigned a value of part_ok = 1.

We generate similar target data for 'pressure_packing', 'temperature_die' and 'pressure_phase2'.

Let's look at a heatmap of the correlations of the random forest model:

![Bildschirmfoto vom 2023-11-02 14-12-43](https://github.com/emefff/Machine-Learing-in-Die-Casting-Production-with-Sklearn/assets/89903493/7dc08bf9-97dd-4051-8504-557925834398)

As the reader may imagine, the possibilities are next to endless with different data, more data, etc.

Due to the artificial nature of the data, the model makes very good predictions near 100% accuracy. This is very likely not the case for real life data. Reasons could be: insufficient training, scrap due to a parameter that is not recorded etc.
The model is later optimized with GridSearchCV. The diagram below shows accuracy and cv_score over number of shots. With the percentages of scrap we set/calculate here, we only need approx. more than 1000 shots to get scores near 100%. Curves of rf_model and rf_tuned are always very close, indicating that the default values of the RFClassifier are already very good. There is not much point in optimizing with these data. As the data varies from run to run, there is more variance between the runs:

![Bildschirmfoto vom 2023-11-02 14-11-16](https://github.com/emefff/Machine-Learing-in-Die-Casting-Production-with-Sklearn/assets/89903493/971277aa-6705-4407-b2ca-35952795cd91)

How can we predict from new data if a part will be scrap or good?

This is done with ML_Die_Casting_open_model_sample_preds.py.
Here we load the data an can make predicitons without looking at the part! Image a machine sending its shot data to this model: during the cooling and handling of the part there is a lot of time (+10s!) for inference. Thus we have enough time to make a prediciton and send the part, if the model predicts 'SCRAP', to a scrap or inspection area. The number of parts that have to be inspected may be reduced to a minimum.
At the moment, this is still only a text output:

![Bildschirmfoto vom 2023-11-01 17-07-21](https://github.com/emefff/Machine-Learing-in-Die-Casting-Production-with-Sklearn/assets/89903493/edd92f9b-6e65-4215-816e-4e652ab05ead)

emefff@gmx.at

