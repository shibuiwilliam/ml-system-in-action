import os
import streamlit as st
import csv
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

LOG_FILE = os.getenv('LOG_FILE', '/tmp/ab_test_log.csv')


@st.cache
def training_data():
    iris = load_iris()
    iris_df = pd.DataFrame(iris['data'], columns=iris.feature_names)
    iris_df['label'] = iris['target']
    iris_df.loc[iris_df['label'] == 0, 'label'] = "setosa"
    iris_df.loc[iris_df['label'] == 1, 'label'] = "versicolor"
    iris_df.loc[iris_df['label'] == 2, 'label'] = "virginica"
    return iris_df


@st.cache
def load_data():
    df = pd.read_csv(LOG_FILE)
    df = df.set_index('job_id')
    return df


@st.cache
def ab_group(df):
    df_a = df[df['ab_test_group'] == 'A']
    df_b = df[df['ab_test_group'] == 'B']
    df_ab = df_a.drop('prediction', axis=1)
    df_ab['id'] = list(range(len(df_ab)))
    df_ab['prediction_a'] = df_a['prediction']
    df_ab['prediction_b'] = df_b['prediction']
    df_ab.loc[df_ab['prediction_a'] == 0, 'prediction_a'] = "setosa"
    df_ab.loc[df_ab['prediction_a'] == 1, 'prediction_a'] = "versicolor"
    df_ab.loc[df_ab['prediction_a'] == 2, 'prediction_a'] = "virginica"
    df_ab.loc[df_ab['prediction_b'] == 0, 'prediction_b'] = "setosa"
    df_ab.loc[df_ab['prediction_b'] == 1, 'prediction_b'] = "versicolor"
    df_ab.loc[df_ab['prediction_b'] == 2, 'prediction_b'] = "virginica"
    return df_ab


def main():
    st.title('ML input data and prediction')

    iris_df = training_data()

    st.subheader('Original data distribution')
    sns.pairplot(iris_df)
    st.pyplot()

    df = load_data()
    df_ab = ab_group(df)
    input_df = df_ab[['id', 'datetime', 'data_0', 'data_1', 'data_2', 'data_3']]

    index_start = st.sidebar.slider('index start', 0, len(df_ab)-500, 10)
    index_add = st.sidebar.slider('index range', 0, 500, 10)

    st.subheader('Input data and prediction records')
    st.subheader(f'Data size: {df_ab.shape}')
    st.dataframe(df_ab)
    sns.pairplot(input_df[['data_0', 'data_1', 'data_2', 'data_3']][index_start:index_start+index_add])
    st.pyplot()

    p0 = plt.plot(input_df['id'][index_start:index_start+index_add], input_df['data_0'][index_start:index_start+index_add], linewidth=1)
    p1 = plt.plot(input_df['id'][index_start:index_start+index_add], input_df['data_1'][index_start:index_start+index_add], linewidth=1)
    p2 = plt.plot(input_df['id'][index_start:index_start+index_add], input_df['data_2'][index_start:index_start+index_add], linewidth=1)
    p3 = plt.plot(input_df['id'][index_start:index_start+index_add], input_df['data_3'][index_start:index_start+index_add], linewidth=1)
    plt.legend((p0[0], p1[0], p2[0], p3[0]), ('data_0', 'data_1', 'data_2', 'data_3'), loc=2)
    st.pyplot()


if __name__ == '__main__':
    main()
