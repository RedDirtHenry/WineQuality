import warnings

warnings.filterwarnings("ignore")
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

from data_processing import split_data

def correlation_features(df, cols):
    numeric_col = df[cols]
    corr = numeric_col.corr()
    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.5:
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features

def lr_model(x_train, y_train):
    # create a fitted model
    x_train_with_intercept = sm.add_constant(x_train)
    lr_mod = sm.OLS(y_train, x_train_with_intercept)
    lr_results = lr_mod.fit()
    return lr_results


def identify_significant_vars(lr, p_value_threshold=0.05):
    # print the p-values
    print(lr.pvalues)
    # print the r-squared value for the model
    print(lr.rsquared)
    # print the adjusted r-squared value for the model
    print(lr.rsquared_adj)
    # identify the significant variables
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    return significant_vars


if __name__ == "__main__":
    win_data = pd.read_csv("/home/jtotiker/Documents/DataScience/Projects/LinReg/data/winsorized_data.csv")
    print(win_data.shape)

    #corr_features = correlation_features(win_data, win_data.columns)
    #print(corr_features)

    cols = ['quality','alcohol','volatile acidity','sulphates','citric acid','density','total sulfur dioxide']

    x_train, x_test, y_train, y_test = split_data(win_data[cols], "quality")

    oversample = RandomOverSampler(random_state=88)

    x_train, y_train = oversample.fit_resample(x_train, y_train)

    data_scaler = StandardScaler()
    data_scaler.fit(x_train)
    x_train = pd.DataFrame(columns=x_train.columns, data=data_scaler.transform(x_train), index=x_train.index)
    x_test = pd.DataFrame(columns=x_test.columns, data=data_scaler.transform(x_test), index=x_test.index)


    lr_results = lr_model(x_train, y_train)
    summary = lr_results.summary()
    print("Train Data", summary)

    predictions = round(lr_results.predict(sm.add_constant(x_test)))

    df = pd.concat([y_test, x_test], axis = 1)
    print(df.head())

    df = pd.concat([predictions.rename("Predicted_Quality"),df], axis=1)
    print(df.columns)

    df_compare = pd.melt(df[["Predicted_Quality","quality"]])

    print(df_compare)

    sns.countplot(data = df_compare, x = df_compare["value"], hue=df_compare["variable"], palette=['#C70039',"#0988B0"]).set(title="Actual v. Predicted Qualities")
