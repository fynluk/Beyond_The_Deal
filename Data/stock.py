
class Stock:
    def __init__(self, name: str):
        self.name = name
        self.prices = self.get_prices()

    def get_prices(self):

        return self.prices

    def __repr__(self):
        return f"Deal(Target: {self.target}, Buyer: {self.buyer}, Date: {self.deal_date})"