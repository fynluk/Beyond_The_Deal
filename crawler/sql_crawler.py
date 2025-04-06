import mysql.connector
import logging
from IPython.core.display_functions import display

def initDatabase(db, stock):
    cursor = db.db.cursor(buffered=True)
    cursor.execute("SHOW TABLES")
    check = False
    for x in cursor:
        if x[0] == stock.ticker:
            check = True
    if check:
        logging.info("Database for Stock: " + stock.ticker + " already exists")
    else:
        logging.info("Database for Stock: " + stock.ticker + " will be created")
        query = "CREATE TABLE `" + stock.ticker + "` (Date VARCHAR(20), Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, Volume FLOAT)"
        cursor.execute(query)

class SqlHandler:
    def __init__(self, config):
        try:
            self.db = mysql.connector.connect(
                host=config.get("sql", {}).get("HOST"),
                user=config.get("sql", {}).get("USER"),
                password=config.get("sql", {}).get("PASSWORD"),
                database=config.get("sql", {}).get("DATABASE"),
            )
        except:
            logging.error("SQL Bridge connection failed")

    def init(self, deals):
        for deal in deals:
            initDatabase(self, deal.buyer)
            initDatabase(self, deal.target)