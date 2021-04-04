import pandas as pd
def get_data():
    data=pd.read_csv("data/heart.csv")
    return data