import sys
from typing import List
import progressbar
import yaml
import logging
from crawler.refinitive_crawler import RefinitivHandler
from crawler.sql_crawler import SqlHandler
from Data.deal import Deal
from Data.stock import Stock
import Math.plot as plt

def load_config(file_path="Configuration/config.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def load_statistics(file_path="Configuration/statistics.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file).get("imaa", [])
    return config

def load_deals(ref, file_path="Configuration/deals.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        dealDict = yaml.safe_load(file).get("Deals")
    deals: List [Deal] = []

    bar = progressbar.ProgressBar(widgets=[
        'Load Deals: ', progressbar.Counter(), '/', str(len(dealDict)), ' ', progressbar.Percentage(), ' ', progressbar.Bar(fill="."), ' ', progressbar.ETA()
    ])
    for x in bar(dealDict):
        stockB = Stock(
            ref.getName(x.get("buyer")),
            x.get("buyer"))
        stockT = Stock(
            ref.getName(x.get("target")),
            x.get("target"))

        deal = Deal(
            stockB,
            stockT,
            x.get("aDate"),
            "0"
        )
        logging.info("Found Deal: " + str(deal))
        deals.append(deal)
    return deals

def main():
    logging.basicConfig(level=logging.INFO, filename="runtime.log", filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Create Statistic Charts")
    data = load_statistics()
    #plt.imaa(data)

    logging.info("Load Configuration")
    config = load_config()

    logging.info("Init Refinitiv Bridge")
    ref = RefinitivHandler(config)

    logging.info("Init SQL Bridge")
    sql = SqlHandler(config)

    logging.info("Loading Deals")
    deals = load_deals(ref)

    # Prüfen, ob 'skip' als Argument übergeben wurde
    skip = len(sys.argv) > 1 and sys.argv[1].lower() == "skip"

    if not skip:
        logging.info("Creating Database Tables")
        sql.init(ref, deals)

        logging.info("Get and Upload Historical Data")
        ref.getPrices(sql)
    else:
        logging.info("Skip mode active – skipping table creation and data upload.")

    logging.info("Create Intervals")
    intervals_buyer: List = []
    intervals_target: List = []

    bar = progressbar.ProgressBar(widgets=[
        'Get Intervals for Ticker: ', progressbar.Counter(), '/', str(len(deals)), ' ', progressbar.Percentage(),
        ' ', progressbar.Bar(fill="."), ' ', progressbar.ETA()
    ])
    for deal in bar(deals):
        intervals_buyer.append(deal.buyer.get_interval(sql, deal.announcement_date, 20, True))
        intervals_target.append(deal.target.get_interval(sql, deal.announcement_date, 20, True))
    #plt.show_interval(intervals_buyer[0], deals[0].buyer)
    #plt.show_interval(intervals_target[0], deals[0].target)
    return_buyer: List = []
    return_target: List = []
    for int in intervals_buyer:
        return_int = (int[20]- int[-20]) / int[-20]
        return_buyer.append(return_int)
    plt.show_returns(return_buyer, deals)
    for int in intervals_target:
        return_int = (int[20]- int[-20]) / int[-20]
        return_target.append(return_int)
    plt.show_returns(return_target, deals)


    logging.info("Calculate Unaffected Share-Price")

    # TODO (Housekeeping?) Is enough data available for every Deal/Stock?
    # TODO (Uneffected SharePrice) Start with the Monte-Carlo-Simulation
    sql.db.close()

if __name__ == "__main__":

    main()
