# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:01:18 2019

@authors: Ankita Chatterjee, Jessica Qiu, Samaneh Zamanifard 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dirpath = os.getcwd()
print("Current Directory is: " + dirpath)
filepath = " "

filepath = input("Enter file path (if the file is kept on run directory press. Enter twice): ")

if not filepath.strip():
    filepath = dirpath
    
#If no Filename given as input, Filename is fixed and provided
filename1 = input('Enter File Name FremontBridge: ')
filename2 = input('Enter File Name BicycleWeather: ')

if not filename1.strip():
    filename1 = "FremontBridge.csv"
if not filename2.strip():
    filename2 = "BicycleWeather.csv"

filename1 = filepath+"\\"+filename1
filename2 = filepath+"\\"+filename2

counts = pd.read_csv(filename1, index_col='Date', parse_dates=True)
weather = pd.read_csv(filename2, index_col='DATE', parse_dates=True)

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

def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)

# temperatures are in 1/10 deg C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

daily['annual'] = (daily.index - daily.index[0]).days / 365.

daily.head()

# Drop any rows with null values
daily.dropna(axis=0, how='any', inplace=True)

daily['SNOW'] = weather['SNOW']
daily['new_temp'] = daily['Temp (C)'] ** 2
daily['TMAX'] = weather['TMAX']
daily['TMIN'] = weather['TMIN']
daily['TMAX_MIN2'] = (daily['TMIN'] * daily['TMAX']) ** 2
daily['new_daylight'] = daily['daylight_hrs'] ** 2
daily['WDF22'] = weather['WDF2'] ** 2
daily['WDF52'] = weather['WDF5'] ** 2
daily['WSF2'] = weather['WSF2']
daily['Mon_daylight2'] = (daily['Mon'] * daily['daylight_hrs']) ** 2
daily['temp_daylight2'] = (daily['daylight_hrs'] * daily['Temp (C)']) ** 2
daily['Wed_daylight2'] = (daily['Wed'] * daily['daylight_hrs']) ** 2
daily['Thu_daylight2'] = (daily['Thu'] * daily['daylight_hrs']) ** 2
daily['Fri_daylight2'] = (daily['Fri'] * daily['daylight_hrs']) ** 2
daily['Sat_daylight2'] = (daily['Sat'] * daily['daylight_hrs']) ** 2
daily['Sun_daylight2'] = (daily['Sun'] * daily['daylight_hrs']) ** 2
daily['annual2'] = daily['annual'] ** 2

column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual', 
                'new_temp', 'SNOW', 'TMAX', 'TMIN','TMAX_MIN2','new_daylight',
                'WDF22','Mon_daylight2','WDF52','WSF2','temp_daylight2', 'Sat_daylight2',
                'Sun_daylight2', 'Fri_daylight2', 'Wed_daylight2', 'Thu_daylight2', 'annual2']

X = daily[column_names]
y = daily['Total']

model = PolynomialFeatures(degree=2)
xPoly = model.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(xPoly, y)

linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(X, y)

#model = LinearRegression(fit_intercept=False)
#model.fit(X, y)
#daily['predicted'] = model.predict(X)

daily['predicted'] = linear_model.predict(X)
daily['predicted'] = poly_reg.predict(xPoly)

daily[['Total', 'predicted']].plot(alpha=0.5);

original_mse = 207632.100662988
new_mse = mean_squared_error(daily['Total'], daily['predicted'])
precision = ((original_mse - new_mse) / original_mse) * 100
#new_mse = round(mean_squared_error(daily['Total'], daily['predicted']), 4)
#precision = round((( (original_mse - new_mse) / original_mse) * 100), 4)

print("Original MSE: ", original_mse)
print("New MSE: ", new_mse)
print("MSE Reduction Percentage: ", precision)
