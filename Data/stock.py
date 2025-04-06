
class Stock:
    def __init__(self, name: str, ticker: str):
        self.name = name
        self.ticker = ticker

    def __repr__(self):
        return f"Stock({self.name}, {self.ticker})"