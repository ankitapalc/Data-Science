# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:01:18 2019

@author: jessicq
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from math import sqrt
from math import log
from sklearn.metrics import mean_squared_error
#from sklearn import preprocessing, normalize
from sklearn.linear_model import Lasso, LinearRegression, Ridge

counts = pd.read_csv(r'C:\Users\palan\OneDrive\Desktop\Clemson Data Science\FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv(r'C:\Users\palan\OneDrive\Desktop\Clemson Data Science\BicycleWeather.csv', index_col='DATE', parse_dates=True)

daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']] # remove other columns

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
	daily[days[i]] = (daily.index.dayofweek == i).astype(float)

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)

#####################################################################
def mean_c(inp_data_x):
	return sum(inp_data_x) / len(inp_data_x)


def standardize(inp_data_x):
	result = np.empty(len(inp_data_x))
	variances = np.linspace(1,len(inp_data_x),len(inp_data_x))
	mean_x = mean_c(inp_data_x)
	for i in range(len(inp_data_x)):
		variances[i] = (inp_data_x[i]-mean_x)**2
	stdev = sqrt(sum(variances)/(len(inp_data_x)))
	for i in range(len(inp_data_x)):
		result[i] = (inp_data_x[i] - mean_x)/stdev
	return result#, mean_x, stdev
#####################################################################



def hours_of_daylight(date, axis=23.44, latitude=47.61):
#"""Compute the hours of daylight for the given date"""
	days = (date - pd.datetime(2000, 12, 21)).days
	m = (1. - np.tan(np.radians(latitude))* np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
	return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)

# temperatures are in 1/10 deg C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
daily['SNOW'] = weather['SNOW']

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

daily['annual'] = (daily.index - daily.index[0]).days / 365.

daily.head()

# Drop any rows with null values
daily.dropna(axis=0, how='any', inplace=True)

daily['SNOW'] = weather['SNOW']
daily['new_temp'] = daily['Temp (C)'] ** 3
daily['TMAX'] = weather['TMAX']
daily['TMIN'] = weather['TMIN']
daily['TMAX_MIN2'] = (daily['TMIN'] * daily['TMAX']) ** 2
daily['new_daylight'] = daily['daylight_hrs'] ** 2
daily['Mon_daylight2'] = (daily['Mon'] * daily['daylight_hrs']) ** 2
daily['temp_daylight2'] = (daily['daylight_hrs'] * daily['Temp (C)']) ** 2
daily['Sat_daylight2'] = (daily['Sat'] * daily['daylight_hrs']) ** 2
daily['Sun_daylight2'] = (daily['Sun'] * daily['daylight_hrs']) ** 2


column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday','daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual','new_temp', 'SNOW', 'TMAX', 'TMIN','TMAX_MIN2','new_daylight','Mon_daylight2','temp_daylight2', 'Sat_daylight2','Sun_daylight2']

X = daily[column_names]
print(X.head())
y = daily['Total']
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X)
daily[['Total', 'predicted']].plot(alpha=0.5)

original_mse = 207632
new_mse = mean_squared_error(daily['Total'], daily['predicted'])
precision = ((original_mse - new_mse) / original_mse) * 100
#new_mse = round(mean_squared_error(daily['Total'], daily['predicted']), 4)
#precision = round((( (original_mse - new_mse) / original_mse) * 100), 4)

print("Original MSE: ", original_mse)
print("New MSE: ", new_mse)
print("MSE Reduction Percentage: ", precision)
