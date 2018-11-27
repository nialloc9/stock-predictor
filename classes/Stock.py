import numpy, quandl
from sklearn.linear_model import LinearRegression

class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = {}

    def fetch_data(self):
        self.data = quandl.get(self.ticker)
        return self.data

    def get_data(self):
        return self.data

    def get_ticker(self):
        return self.ticker
