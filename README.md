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

    
