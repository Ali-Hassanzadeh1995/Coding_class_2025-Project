from bs4 import BeautifulSoup
import requests
import pandas as pd


def Symbols_df_maker():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {"User-Agent": "AliWikiBot/1.0 (https://github.com/ali)"}

    page = requests.get(url, headers=headers)

    soup = BeautifulSoup(page.text, "html")
    table = soup.find_all("table")[0]
    word_titles = table.find_all("th")
    word_titles_list = [title.text.strip() for title in word_titles]
    df = pd.DataFrame(columns=word_titles_list)
    row_datas = table.find_all("tr")
    for data in row_datas[1:]:
        row_data = [d.text.strip() for d in data if d.text.strip() != ""]
        # print(row_data)
        lenght = len(df)
        df.loc[lenght] = row_data
    df.to_csv("Symbols.csv", index=False)


# Turn it to a class that fetches tables form any wiki page!!
