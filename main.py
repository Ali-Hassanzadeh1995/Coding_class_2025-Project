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
if test.lower() == test:
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
