# description: Project 1

from functions import *

# imports the random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ============= Importing the data =====================================

# reads a csv file to panda dataframe
df = import_data()

# ============= Processing the data =====================================

df = process_data(df)

# ============= 1- Plotting mean and std =====================================

# plots multiple plots in the same figure
plot_all_means_and_std(df, x_names, y_name)

# ============= 2- ID3, cross val., ROC, AUC ROC and confusion matrix ===================

pipe_best_feats = []

# creates a decision tree classifier using the ID3 algorithm with cross validation
dt = DecisionTreeClassifier(criterion="entropy")

# collects the best features from the cross validation
pipe_best_feats.append(pipeline(dt, df))

# ============= 3- random forest m=9, cross val., ROC, AUC ROC and confusion matrix ===================

# uses the random forest algorithm with cross validation
rf = RandomForestClassifier(n_estimators=100)

# collects the best features from the cross validation
pipe_best_feats.append(pipeline(rf, df, max_features="no limit"))

# ============= 4- random forest m=3, cross val., ROC, AUC ROC and confusion matrix ===================

max_features = 3

# uses the random forest algorithm with cross validation
rf = RandomForestClassifier(n_estimators=100, max_features=max_features)

# collects the best features from the cross validation
pipe_best_feats.append(pipeline(rf, df, max_features=str(max_features)))

pipe_best_feats.sort(lambda x: x[1])

# prints the best variables in the terminal
print(
    "the 2 most influential variables of the best model: \n"
    + str(pipe_best_feats[0][0][0])
)

# ============= 5- random forest m=3, cross val., ROC, AUC ROC and confusion matrix ===================
