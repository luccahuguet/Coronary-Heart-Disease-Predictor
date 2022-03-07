#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

x_names = "sbp,tobacco,ldl,adiposity,famhist,obesity,alcohol,age".split(",")
y_name = "chd"


def import_data():
    fname = "SA_heart.csv"
    df = pd.read_csv(fname)
    return df


def process_data(df):

    # drops the id column
    df = df.drop(columns=["id"])
    df = df.drop(columns=["typea"])

    # converts categorical values to numerical values
    df = convert_cat_to_num(df)
    return df


# converts "fam_hist" to numerical values and returns a new dataframe
def convert_cat_to_num(dataframe):
    fam_hist_dict = {"Present": 1, "Absent": 0}
    dataframe["famhist"] = dataframe["famhist"].map(fam_hist_dict)
    return dataframe


def plot_all_means_and_std(df, x_names, y_name):

    _, axis = plt.subplots(3, 3)

    for i, var in enumerate(x_names):
        df_chd = df.groupby(y_name).agg(["mean", "std"])

        chd = df_chd[var]
        chd.plot(
            kind="barh",
            y="mean",
            legend=False,
            title="Average " + var,
            xerr="std",
            ax=axis[i // 3, i % 3],
        )

    plt.show()


def get_best_2_params(cv_dict, X_test, y_test):

    feature_imp_all = []
    for est in cv_dict["estimator"]:

        roc = roc_auc_score(y_test, est.predict_proba(X_test)[:, 1])
        feature_imp = pd.Series(est.feature_importances_, index=x_names).sort_values(
            ascending=False
        )
        feature_imp_all.append((feature_imp[:2], roc))

        # print(f"Feature importance: \n{feature_imp}")
    feature_imp_all.sort(key=lambda x: x[1], reverse=True)
    return feature_imp_all


# feats[0][0], feats[1][0]


def plot_AUC_ROC(cv_dict, X_test, y_test, max_features=None):

    for estimator in cv_dict["estimator"]:
        fpr, tpr, _ = roc_curve(y_test, estimator.predict_proba(X_test)[:, 1])
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1]):.3f}",
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "AUC ROC"
            + " - "
            + estimator.__class__.__name__
            + " \n max_features = "
            + str(max_features)
        )
        plt.legend(loc="lower right")
    plt.show()


# these are shown in the terminal
def calc_confusion_matrix(cv_dict, X_test, y_test, max_features=None):
    y_pred = cv_dict["estimator"][np.argmax(cv_dict["test_score"])].predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    message = f"Confusion matrix \n"
    print(message, cm, end="\n\n")


def pipeline(stimator, df, max_features=None):

    title = f"\n\n{stimator.__class__.__name__}"

    if max_features is not None:
        title += f"- with max_features: {max_features} \n"

    print(title)

    X_train, X_test, y_train, y_test = train_test_split(
        df[x_names], df[y_name], test_size=0.1, random_state=1
    )

    # cross validation and fitting
    cv_dict = cross_validate(
        stimator,
        X_train,
        y_train,
        cv=10,
        return_estimator=True,
        return_train_score=True,
    )

    feats = get_best_2_params(cv_dict, X_test, y_test)[:1]

    # plotting the AUC curve
    plot_AUC_ROC(cv_dict, X_test, y_test, max_features)

    # plots the confusion matrix
    calc_confusion_matrix(cv_dict, X_test, y_test, max_features)

    return feats
