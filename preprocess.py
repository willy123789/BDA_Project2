import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns## 資料視覺化
import plotly.express as ex## 資料視覺化
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib

matplotlib.rc('font', family='Microsoft JhengHei')

class Data():
    def add_feature(df, path, col_name):
        # add feature from other csv file
        df_new = pd.read_csv(path)
        target_col = df_new[col_name]
        df[col_name] = target_col
        
        return df

class Station():
    def date_convert(df): 
        # convert data format to %Y/%m/%d
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime('%Y/%m/%d')
        
        # move date column to the first column
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('date')))
        df = df.reindex(columns= cols)
        return df
        
    def clean_features(df):
        # drop unnecessary columns
        try:
            df.drop(['ObsTime', 'StnPresMaxTime', 'StnPresMinTime','T Max Time','T Min Time','RHMinTime','WGustTime','T Max','T Min','Temperature','GloblRad'], axis=1, inplace=True)
        except:
            df.drop(['ObsTime', 'StnPresMaxTime', 'StnPresMinTime','T Max Time','T Min Time','RHMinTime','WGustTime','T Max','T Min','Temperature'], axis=1, inplace=True)
        return df
    
    def name_columns(df, name):
        # add name to each column but date column doesn't need to add
        df.columns = [name + '_' + col if col != 'date' else col for col in df.columns]
        return df
    
class Visualization():
    def plot(df, title):
        # plot the data
        df.plot(figsize=(20,10), title=title)
        plt.show()
        
    def plot_all(df, title):
        # plot the data
        df.plot(figsize=(20,10), title=title)
        plt.show()
        
    def plot_each(df, title):
        # plot the data
        df.plot(figsize=(20,10), title=title)
        plt.show()
        
    def plot_all_each(df, title):
        # plot the data
        df.plot(figsize=(20,10), title=title)
        plt.show()
        
    def plot_distribution(df):
        # plot all features distribution
        fig = plt.figure(figsize=(20, 20))
        for i in range(len(df.columns)):
            ax = fig.add_subplot(5, 5, i+1)
            sns.distplot(df.iloc[:, i], ax=ax)
        plt.tight_layout()
        plt.show()
        
    def plot_correlation(df):
        # plot the correlation between features
        plt.figure(figsize=(20, 20))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.show()
        
    def plot_regression(y_test, y_pred):
        # Assuming y_test and y_pred are your actual and predicted values

        # Create scatter plot
        sns.set_style("darkgrid")
        plt.figure(figsize=(7, 7))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')

        # Add regression line
        axes = plt.gca()
        m, b = np.polyfit(y_test.iloc[:,0], y_pred[:,0], 1)
        X_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
        plt.plot(X_plot, m*X_plot + b, '-', color='red')
        plt.show()