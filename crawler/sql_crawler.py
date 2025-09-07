import mysql.connector
import logging
import pandas as pd
import pandas._libs.missing
from IPython.core.display_functions import display
import progressbar
from numpy.distutils.misc_util import quote_args


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
        query = "CREATE TABLE `" + stock.ticker + "` (Date VARCHAR(20), Open DECIMAL(16, 2), High DECIMAL(16, 2), Low DECIMAL(16, 2), Close DECIMAL(16, 2), Volume DECIMAL(16, 2), PRIMARY KEY (Date))"
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
        bar = progressbar.ProgressBar(widgets=[
            'Create DB Tables for each Deal: ', progressbar.Counter(), '/', str(len(deals)), ' ', progressbar.Percentage(), ' ', progressbar.Bar(fill="."), ' ', progressbar.ETA()
        ])
        for deal in bar(deals):
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

    def uploadData(self, ticker, tupel):
        # Check whether Data exists for the date
        cursor = self.db.cursor(buffered=True)
        # If value has type pd.NA use Value 0.0
        def sanitize(value):
            if isinstance(value, pandas._libs.missing.NAType):
                return 0.0
            elif isinstance(value, type(None)):
                return 0.0
            else:
                return value

        # Bereite Daten für das Insert vor
        sanitized_data = []
        for (date, open, high, low, close, vol) in tupel:
            sanitized_row = (
                str(date),  # Sicherstellen, dass Datum ein String ist
                float(sanitize(open)),
                float(sanitize(high)),
                float(sanitize(low)),
                float(sanitize(close)),
                int(float(sanitize(vol)))
            )
            sanitized_data.append(sanitized_row)

        # Query mit Platzhaltern
        query = f"INSERT IGNORE INTO `{ticker}` (Date, Open, High, Low, Close, Volume) VALUES (%s, %s, %s, %s, %s, %s);"

        # Jetzt mit executemany alle Datensätze einfügen
        #print(query)
        #print(sanitized_data)
        cursor.executemany(query, sanitized_data)
        self.db.commit()

    def get_interval(self, ticker, aDate, interval, bool_abs):
        cursor = self.db.cursor()

        # Convert aDate from String to pd.Timestamp
        if isinstance(aDate, pd.Timestamp):
            aDate = aDate.strftime("%Y-%m-%d")

        # Get Prices for Ticker from Database
        query = f"""
                SELECT Date, Open FROM `{ticker}`
                ORDER BY Date
            """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        # Convert SQL-Query to DataFrame
        df = pd.DataFrame(rows, columns=["Date", "Open"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # Find aDate and get Index
        try:
            center_index = df.index[df["Date"] == pd.to_datetime(aDate).date()][0]
        except IndexError:
            logging.ERROR(f"Datum {aDate} nicht in Datenbank für Ticker {ticker} enthalten.")

        start_index = center_index - interval
        end_index = center_index + interval + 1

        # Create Interval with Values (abs or rel)
        subset = df.iloc[start_index:end_index].reset_index(drop=True)
        relative_position = interval * -1
        if bool_abs:
            result_abs = {}
            for index, row in subset.iterrows():
                result_abs[relative_position] = row['Open']
                relative_position = relative_position + 1
            try:
                test = result_abs[20]
            except:
                logging.warning("Interval not long enough for ticker: " + ticker)
                print("Aktuell: " + str(result_abs))
                last_key, last_value = list(result_abs.items())[-1]
                result_abs[20] = last_value
                print("New: " + str(result_abs))

            return result_abs
        else:
            zero_value = df.iloc[center_index]["Open"]
            result_rel = {}
            for index, row in subset.iterrows():
                result_rel[relative_position] = row['Open'] / zero_value
                relative_position = relative_position + 1
            return result_rel