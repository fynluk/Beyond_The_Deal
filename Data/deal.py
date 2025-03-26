from datetime import date

class Deal:
    def __init__(self, target: str, buyer: str, announcement_date: date, deal_date: date):
        self.target = target
        self.buyer = buyer
        self.announcement_date = announcement_date
        self.deal.date = deal_date

    def __repr__(self):
        return f"Deal(Target: {self.target}, Buyer: {self.buyer}, Date: {self.deal_date})"