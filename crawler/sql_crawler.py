import mysql.connector
import logging

import pandas._libs.missing
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
        query = "CREATE TABLE `" + stock.ticker + "` (Date VARCHAR(20), Open DECIMAL(16, 2), High DECIMAL(16, 2), Low DECIMAL(16, 2), Close DECIMAL(16, 2), Volume DECIMAL(16, 2))"
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

    def init(self, ref,  deals):
        for deal in deals:
            initDatabase(self, deal.buyer)
            initDatabase(self, deal.target)

    def getTickers(self):
        cursor = self.db.cursor(buffered=True)
        cursor.execute("SHOW TABLES")
        tickers = [t[0] for t in cursor.fetchall()]
        tickersToFetch = []
        for ticker in tickers:
            #cursor.execute("SELECT * FROM " + ticker)
            if True:
                # TODO Only return tickers where data is missing
                tickersToFetch.append(ticker)
        return tickersToFetch

    def uploadData(self, ticker, date, open, high, low, close, vol):
        # Check whether Data exists for the date
        cursor = self.db.cursor(buffered=True)
        check_query = f"SELECT 1 FROM `{ticker}` WHERE `Date` = %s LIMIT 1"
        cursor.execute(check_query, (date,))
        exists = cursor.fetchone()
        if exists:
            # Datapoint exists
            pass
        else:
            # If value has type pd.NA use Value 0.0
            def sanitize(value):
                return 0.0 if type(value) == pandas._libs.missing.NAType else value

            # Werte bereinigen
            open = sanitize(open)
            high = sanitize(high)
            low = sanitize(low)
            close = sanitize(close)
            vol = sanitize(vol)

            insert_query = f"""
                        INSERT INTO `{ticker}` (Date, Open, High, Low, Close, Volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
            cursor.execute(insert_query, (date, open, high, low, close, vol))
            self.db.commit()