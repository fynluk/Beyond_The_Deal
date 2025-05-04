import logging

class Stock:
    def __init__(self, name: str, ticker: str):
        self.name = name
        self.ticker = ticker

    def __repr__(self):
        return f"Stock({self.name}, {self.ticker})"

    def get_interval(self, sql, aDate, interval, bool_abs):
        logging.info("Fetching Interval for Stock: " + self.ticker + " with length: " + str(interval*2))
        result = sql.get_interval(self.ticker, aDate, interval, bool_abs)
        return result