from datetime import date
from Data.stock import Stock

class Deal:
    def __init__(self, buyer: Stock, target: Stock, announcement_date: str, deal_date: str):
        self.target = target
        self.buyer = buyer
        self.announcement_date = announcement_date
        self.deal_date = deal_date

    def __repr__(self):
        return f"Deal(Target: {self.target}, Buyer: {self.buyer}, Announcement-Date: {self.announcement_date}, Deal-Date: {self.deal_date})"