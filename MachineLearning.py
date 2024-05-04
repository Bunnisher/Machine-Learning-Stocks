from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
from pandas import *
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import yfinance as yf
import math, datetime
from sklearn import model_selection, preprocessing, svm
from sklearn.linear_model import LinearRegression
yf.pdr_override()

root = Tk()
root.title("Bunnisher's ML Tool")
root.geometry("300x400")
root.configure(background='gray23')

cap_image = ImageTk.PhotoImage(Image.open("bunny2.jpg"))
cap_image_label = Label(image=cap_image)
cap_image_label.grid(row=0, column=0)

stockChoice = Entry(root, width=50)
stockChoice.grid(row=2, column=0, pady=10, padx=10)



def startStockSearch():
    data = pdr.get_data_yahoo(stockChoice.get(), start= "2020-01-22", stop= "2020-09-22", interval='1d')

    df = data

    df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100.0
    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

    forecast_col = 'Adj Close'
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.01*len(df)))
    print(forecast_out)

    df['label'] = df[forecast_col].shift(-forecast_out)
    df.dropna(inplace=True)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

    df['Adj Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

    #Date, Close = np.loadtxt(data, delimiter=',', unpack=True)
'''
    data['Adj Close'].plot()
    plt.xlabel('Date')
    plt.ylabel('Adjusted')
    plt.title('Bunnisherz Stock Graph')
    plt.style.use('dark_background')
    plt.grid(color='w', linestyle='solid')
    plt.legend()
    plt.show()



def write_it():
    data = pdr.get_data_yahoo(stockChoice.get(), start= "2020-09-18", stop= "2020-09-18", interval='1m')
    df = pd.DataFrame(data)

    df2 = df['Adj Close']
    
    df2.to_csv('Stocky.csv')
 '''   

startIt = Button(root, text="Start Program",bg='yellow2', command=lambda: startStockSearch())
startIt.grid(row=1, column=0, pady=10)
'''
writeIt = Button(root, text="Write it", command=lambda: write_it())
writeIt.grid(row=3, column=0, pady=10)
'''
button4 = Button(root, text="By Rick", bg='goldenrod2', command=root.destroy)
button4.grid(row=15, column=0, pady=10)


root.mainloop()