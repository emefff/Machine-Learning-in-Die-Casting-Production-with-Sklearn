# Machine-Learing-in-Die-Casting-Production-with-Sklearn

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
![Bildschirmfoto vom 2023-10-31 13-32-03](https://github.com/emefff/Machine-Learing-in-Die-Casting-Production-with-Sklearn/assets/89903493/bd16ed85-eb55-4a3e-9eca-089563d38a88)

As we generate Gaussian values around 4000mm/s (a reasonable value) we get a EXP(-X**2) like distribution of data. Real-life data may have different distributions. Here, we only use the data for training the model.
We also need to create data artificially, that is, we have to tell the model which values of v_phase2 (or any of the other independent parameters) does not result in a good part. At a real machine, reasons can be anything related to the hydraulics of the machine. For example: a valve could be damaged, leading to a very low shot speed v_phase2.
