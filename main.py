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
    deals = []
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
    logging.info("Init Refinitive Bridge")
    ref = RefinitivHandler(config)
    logging.info("Init SQL Bridge")
    sql = SqlHandler(config)
    logging.info("Loading Deals")
    deals = load_deals(ref)
    logging.info("Creating Database Tables")
    sql.init(ref, deals)
    logging.info("Get and Upload Historical Data")
    ref.getPrices(sql)

if __name__ == "__main__":
    #display(test.data.df)
    #print(list(test.data.df.columns.values))
    #print(test.data.df.index)
    #print(test.data.df.iloc[0].name.day)
    #print(test.data.df.iloc[0].name.month)
    #print(test.data.df.iloc[0].name.year)
    #print(type(test.data.df.iloc[0].name))
    #print(test.data.df.to_dict())))
    #print(yaml.dump(test.data.df.to_dict(), default_flow_style=False))
    #rdp.close_session()
    main()
