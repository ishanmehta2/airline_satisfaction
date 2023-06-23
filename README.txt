There are four .py files contained in this folder

1. data_cleaning.py: Python file for pre-processing training and testing datasets. Run code with command “python3 data_cleaning.py” and make sure original datasets are stored in “./data/”. 

2. log_reg.py: Python file containing implementation of logistic regression from scratch (no libraries except pandas and numpy). Run code with command “python3 log_reg.py,” but takes multiple hours to run. The code will output the accuracy of the model and correctly classified test instances by class. 
 
3. naiveBayes.py: Python file containing implementation of Naïve Bayes from scratch (no libraries except pandas and numpy). Run code with command “python3 naiveBayes.py” (takes ~1 min). Make sure “trained_cleaned.csv” and “test_cleaned.csv” are stored in “./data/”. Can choose to remove features by including their names in “excluded_features” list on line 78. The code will output accuracy of the model. 

4. kcluster.py: Python file containing implementation of K-means clustering using scikit-learn package. Need to install scikit-learning using pip. Run code with command “python3 kcluster.py” (takes ~1 min). Make sure “trained_cleaned.csv” and “test_cleaned.csv” are stored in “./data/”. Can choose to remove features by including their names on lines 14 and 15. The code will output the accuracy of the model. 
