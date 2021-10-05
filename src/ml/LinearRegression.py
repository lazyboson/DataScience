import math
import pandas as pd
import quandl, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt
import time
import pickle

style.use('ggplot')
df = quandl.get('WIKI/GOOG')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0;
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0;
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecasst_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.1 * len(df)))
df['label'] = df[forecasst_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['label'])
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=20)
clf = LinearRegression()
clf.fit(X_train, y_train)
# pickle the classifier n order to save time overhead
with open('linear_regression.pickle', 'w+') as f:
    pickle.dump(clf, f)

pickle_in = open('linear_regression.pickle', 'r')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print type(last_date)
last_unix = time.mktime(datetime.date(last_date.year, last_date.month, last_date.day).timetuple())
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.date.fromtimestamp((next_unix))
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
