import sys
from typing import List
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

def load_deals(ref, file_path="Configuration/deals.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        dealDict = yaml.safe_load(file).get("Deals")
    deals: List [Deal] = []
    for x in dealDict:
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
    int_5_buyer = deals[0].buyer.get_interval(sql, deals[0].announcement_date, 5, False)
    plt.show_interval(int_5_buyer, deals[0].buyer)
    int_5_target = deals[0].target.get_interval(sql, deals[0].announcement_date, 5, False)
    plt.show_interval(int_5_target, deals[0].target)

    logging.info("Calculate Unaffected Share-Price")

    # TODO (Housekeeping?) Is enough data available for every Deal/Stock?
    # TODO (Uneffected SharePrice) Start with the Monte-Carlo-Simulation
    sql.db.close()

if __name__ == "__main__":
    #rdp.close_session()
    main()
