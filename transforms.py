import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def drop_record_all_nans(df):
    return df.dropna(how="all", axis=0)


def fillna(df):
    df.fillna(value={
        'age': df.age.mean(),
        'height': df.height.mean(),
        'gender': df.gender.mode()[0],
        'weight': df.weight.mean(),
        'eye_color': df.eye_color.mode()[0],
        'salary': df.salary.mean(),
    }, inplace=True)
    return df


def fill_noisy_data(df):
    df.loc[df.eye_color.astype('str').str.isnumeric(), 'eye_color'] = df.eye_color.mode()[0]
    return df


def change_data_type(df):
    df.height = round(df.height)
    df = df.astype({'age': 'int'})
    return df


def one_hot_encoder(df, column):
    return pd.get_dummies(df, columns=column)


def label_encoding(df, columns):
    le = LabelEncoder()
    for column in columns:
        df[column] = le.fit_transform(df[column])
    return df


def k_bins_discretizer(df, columns):
    dis = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    for column in columns:
        df[column] = dis.fit_transform((df[[column]]))
    return df


def drop_column(df, columns):
    for col in columns:
        df.drop(col, axis=1, inplace=True)
    return df


def check_outlier_column_by_plots(df, columns):
    fig = px.box(df, y=columns)
    fig.show()


def remove_weight_outliers(df, min_w, max_w):
    df = pd.DataFrame(df)
    df = df[(df['weight'] >= min_w) & (df['weight'] <= max_w)]
    return df


def min_max_scaler(df, columns):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = columns
    return df


def standard_scaler(df,columns):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = columns
    return df
