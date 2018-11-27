from classes.Predictor import Predictor


def main():

    ticker = input("Please enter a stock ticker symbol: ")

    predictor = Predictor()

    predictor.set_stock(ticker)

    predictor.train()

    predictor.predict()

    predictor.plot_graph()


main()
