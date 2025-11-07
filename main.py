import pandas as pd
from SP500_Symbol_checker import SP500_Symbol_checker

""" SYMBOLS """
# N =: number of

N = input("pleas enter the number of the stocks:")
while type(N) != int:
    try:
        N = int(N)
    except:
        print("The entry is not an integer!")
        N = input("pleas enter as integer as the number of the stocks:")
# print(N)

# Creating list
test = input(
    f"If you want to test the program you can use enter 'test' to a lsit of stocks with {N} element randomly made for you. if not enter'N'."
)
if test.lower() == "test":
    Set_stocks = SP500_Symbol_checker(None, N).Symbols_random_gen()
else:
    print(
        "Pleas enter the symboles of stoks that you want to be under consideration.\n"
    )
    counter = 0
    Set_stocks = set()
    for i in range(N):
        while True:
            counter += 1
            temp = str(input(f"pleas enter the {i+1}th symbol:"))
            if SP500_Symbol_checker(temp, 0).Symbols_check():
                Set_stocks.add(temp.upper())
                break
            else:
                print(f"{temp.upper()} isn't a valid symbol! Try again.")
            if counter == 3:
                print(f"First check your symbols! See you.")
                break
print(Set_stocks)

"""Start date/ End date"""
List_stocks = Set_stocks
import yfinance as yf

# lower bound of Start Date
min_date = []
for ticker in List_stocks:
    ticker_temp = yf.Ticker(ticker)
    hist = ticker_temp.history(period="max")
    print(f"The lowest valid date for {ticker} is {hist.index.min()}")
    hist = str(hist.index.min())
    hist = str(hist[:9])
    min_date.append([int(hist[0:4]), int(hist[5:7]), int(hist[8:])])
print(min(min_date))

# S_date End_date

from datetime import date

today = str(date.today())
print(f"Note that Format of date is YYYY-MM-DD!, like {today}")
temp = []
temp.append(int(today[0:4]))
temp.append(int(today[5:7]))
temp.append(int(today[8:]))
today = temp[:]

print("Given the Start data of each stock enter a valid start data!")
# tickers = {'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'IBM'}

min_dates = []
for ticker in List_stocks:
    ticker_temp = yf.Ticker(ticker)
    hist = ticker_temp.history(period="max")
    # print(f"The lowest valid date for {ticker} is {hist.index.min()}")
    hist = str(hist.index.min())
    hist = str(hist[:9])
    min_dates.append([int(hist[0:4]), int(hist[5:7]), int(hist[8:])])

max_date = max(min_dates)
print(
    f"Given your ticker(s) list the minimum valid date is {max_date[0]}-{max_date[1]}-{max_date[2]}."
)


def data_format_validation(date):

    for i in range(len(date)):
        if i in [4, 7]:
            if date[i] != "-":
                return False
        else:
            try:
                int(date[i])
            except:
                return False
    if int(date[8] + date[9]) > 31 or int(date[5] + date[6]) not in range(0, 13):
        return False

    return True


S_date = str(input(f"Enter Start Date:"))
temp = S_date

if data_format_validation(S_date):
    temp = []
    temp.append([int(S_date[0:4]), int(S_date[5:7]), int(S_date[8:])])
    S_date = temp[0][:]
    print(S_date)
    if S_date < max_date:
        print(
            f" Entered data is not in valid range. Start data must be grater than equal {max_date[0]}-{max_date[1]}-{max_date[2]}"
        )
else:
    print("Entered data is not in accepteable format!")

if S_date[1] < 10:
    S_date[1] = "0" + str(S_date[1])
if S_date[2] < 10:
    S_date[2] = "0" + str(S_date[2])
S_date = (
    str(S_date[0]) + "-" + str(S_date[1]) + "-" + str(S_date[2])
)  # date is changed to string

E_date = str(input(f"Enter End Date:"))

if data_format_validation(E_date):
    temp = []
    temp.append([int(E_date[0:4]), int(E_date[5:7]), int(E_date[8:])])
    E_date = temp[0][:]
    print(E_date)
    if E_date > today:
        print(
            f" Entered data is not in valid range. Start data must be less than equal {today[0]}-{today[1]}-{today[2]}"
        )
else:
    print("Entered data is not in accepteable format!")

if E_date[1] < 10:
    E_date[1] = "0" + str(E_date[1])
if E_date[2] < 10:
    E_date[2] = "0" + str(E_date[2])

E_date = (
    str(E_date[0]) + "-" + str(E_date[1]) + "-" + str(E_date[2])
)  # date is changed to string

# 1m - 7days, 2m, 5m, 15m, 30m, 60m, 90m - 60 days, 1d(defult), 5d, 1wk, 1mo, 3mo -full history
intervals = [
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]
print("Acceptabale intervals are in the following form:")
print("\n one minutes (maximum duration must be 7 days!): 1m")
print(
    "\n two, five, fifteen, therty, sixty and ninty minutes (maximum duration must be 60 days!): 2m, 5m, ..., 90m"
)
print("\n five day (without any ristriction on duration): 5d ")
print("\n one weak (without any ristriction on duration): 1wk ")
print("\n one or three month (without any ristriction on duration): 1mo, 3mo ")

interval = input("\n Enter your suitable onterval:")

if not interval in intervals:
    print("\n Entered interval is not acceptable!")
else:
    print(f"\n Entered interval is {interval}.")


data = yf.download(
    List_stocks, start=S_date, end=E_date, interval=interval, auto_adjust=False
)
data.head()
DF = data["Adj Close"]

import numpy as np

# DF_simple_retrun = DF/DF.shift(1)-1
DF_simple_retrun = DF.pct_change()
print(DF_simple_retrun)
DF_log_retrun = np.log(DF / DF.shift(1))
DF_log_retrun

DF_simple_mean = DF_simple_retrun.mean()
DF_simple_volatility = DF_simple_retrun.std()
print(DF_simple_mean)
print(DF_simple_volatility)

import pandas as pd

# time_frame and step_time
len_day = int(
    (np.datetime64(E_date) - np.datetime64(S_date)) / np.timedelta64(1, "D")
) // int(interval[0])
print(f"Given the lenght of your data, {len_day}, enter a time_frame and a time_step.")

time_frame = int(input("Enter time_frame:"))
time_step = int(input("Enter time_step:"))
residual = (len(DF) - time_frame) % time_step
steps = (len(DF) - time_frame) // time_step
df_simple_mean = pd.DataFrame(columns=DF.columns.tolist(), index=range(0, steps))
df_simple_volatility = pd.DataFrame(columns=DF.columns.tolist(), index=range(0, steps))
df_log_retrun = pd.DataFrame(columns=DF.columns.tolist(), index=range(0, steps))
# list(DF.columns) == DF.columns.tolist()
# print(df_simple_mean.head())

for i in range(0, steps):
    Temp = DF.iloc[i * time_step + residual : i * time_step + time_frame]
    # print(temp.head())
    df_log_retrun.iloc[i] = [
        np.log(Temp.iloc[j, -1] / Temp.iloc[j, 1]) for j in range(0, Temp.shape[1])
    ]
    Temp = Temp.pct_change()
    temp = Temp.mean()
    # print(list(temp.iloc[:]))
    df_simple_mean.iloc[i] = list(temp.iloc[:])
    temp = Temp.std()
    df_simple_volatility.iloc[i] = list(temp.iloc[:])
print(df_log_retrun)
# print(df_simple_mean)
# print(df_simple_volatility)
import matplotlib.pyplot as plt

df_log_retrun.plot()
plt.show()

df_simple_mean.plot()
plt.show()

df_simple_volatility.plot()
plt.show()
