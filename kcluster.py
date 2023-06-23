import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier # need to install sklearn package: pip install -U scikit-learn
from sklearn.impute import KNNImputer

def main():
    path = "./data/"
    train_df = pd.read_csv(path + "train_cleaned.csv")
    test_df = pd.read_csv(path + "test_cleaned.csv")
    train_df.tail(-1)
    train_df = train_df.dropna()
    test_df.tail(-1)
    test_df = test_df.dropna()
    test_df = test_df.drop(['Gender','Departure/Arrival time convenient','Gate location','Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1)
    train_df = train_df.drop(['Gender','Departure/Arrival time convenient','Gate location','Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1)
    y_val = train_df.iloc[:,-1:]
    x_val = train_df.iloc[:,0:-1]
    y_val_t = test_df.iloc[:,-1:]
    x_val_t = test_df.iloc[:,0:-1]
    x_val.head()
    k_means = KNeighborsClassifier(n_neighbors=15)
    k_means.fit(x_val, y_val.values.ravel())
    print(k_means.score(x_val_t, y_val_t.values.ravel()))

if __name__ == '__main__':
    main()