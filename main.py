import os
import sys
import importlib
import refinitiv.dataplatform as rdp
import yaml
import logging
import json
from IPython.display import display
from crawler.refinitive_crawler import RefinitivHandler
from crawler.sql_crawler import SqlHandler

def load_config(file_path="Configuration/config.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def main():
    logging.basicConfig(level=logging.INFO, filename="runtime.log", filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Load Configuration")
    config = load_config()
    logging.info("Init Refinitive Bridge")
    ref = RefinitivHandler(config)
    logging.info("Init SQL Bridge")
    sql = SqlHandler(config)


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
