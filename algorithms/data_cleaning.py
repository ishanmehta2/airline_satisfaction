import pandas as pd

path = "./data/"
train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")

train_df.head()

test_df.head()

train_df.columns

to_drop = ['Unnamed: 0', 'id']
train_df_cleaned = train_df.drop(to_drop, inplace=False, axis=1)
test_df_cleaned = test_df.drop(to_drop, inplace=False, axis=1)

train_df_cleaned.columns

# Male = 1, Female = 0
train_df_cleaned['Gender'] = train_df_cleaned['Gender'].replace(['Male', 'Female'], [0, 1])
test_df_cleaned['Gender'] = train_df_cleaned['Gender'].replace(['Male', 'Female'], [0, 1])

# For Class
# Eco = 0, Eco Plus = 1, Business = 2
train_df_cleaned['Class'] = train_df_cleaned['Class'].replace(['Eco', 'Eco Plus', 'Business'], [0, 1, 2])
test_df_cleaned['Class'] = test_df_cleaned['Class'].replace(['Eco', 'Eco Plus', 'Business'], [0, 1, 2])

# satisfaction
# not satisfied = 0, satisfied = 1
train_df_cleaned['satisfaction'] = train_df_cleaned['satisfaction'].replace(["neutral or dissatisfied", "satisfied"], [0, 1])
test_df_cleaned['satisfaction'] = test_df_cleaned['satisfaction'].replace(["neutral or dissatisfied", "satisfied"], [0, 1])

# Type of Travel
# Personal Travel = 0, Business travel = 1
train_df_cleaned['Type of Travel'] = train_df_cleaned['Type of Travel'].replace(["Personal Travel", "Business travel"], [0, 1])
test_df_cleaned['Type of Travel'] = test_df_cleaned['Type of Travel'].replace(["Personal Travel", "Business travel"], [0, 1])

# Customer Type
# Loyal Customer = 0, disloyal Customer = 1
train_df_cleaned['Customer Type'] = train_df_cleaned['Customer Type'].replace(["Loyal Customer", "disloyal Customer"], [0, 1])
test_df_cleaned['Customer Type'] = test_df_cleaned['Customer Type'].replace(["Loyal Customer", "disloyal Customer"], [0, 1])

# remove NA values
train_df_cleaned = train_df_cleaned.dropna()
test_df_cleaned = test_df_cleaned.dropna()

train_df_cleaned.head()
test_df_cleaned.head()

train_df_cleaned.to_csv(path + "train_cleaned.csv", index = False)
test_df_cleaned.to_csv(path + "test_cleaned.csv", index = False)

