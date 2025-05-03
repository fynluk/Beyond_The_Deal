import refinitiv.dataplatform as rdp
import logging
import datetime
from IPython.core.display_functions import display


class RefinitivHandler:
    def __init__(self, config, ):
        try:
            rdp.open_platform_session(
                config.get("refinitiv", {}).get("APP-ID"),
                rdp.GrantPassword(
                    config.get("refinitiv", {}).get("USER"),
                    config.get("refinitiv", {}).get("PASSWORD")
                )
            )
        except:
            logging.error("Refinitive Bridge connection failed")

    def getName(self, ticker):
        name = str()
        try:
            name = rdp.convert_symbols(ticker).loc[ticker, "DocumentTitle"].split(",", 1)[0]
        except:
            logging.warning("Ticker may be wrong: " + ticker)
        if name is None:
            logging.error("Name from ticker " + ticker + " not found")
            return "N/A"
        else:
            return name

    def getPrices(self, sql):
        tickers = sql.getTickers()
        for ticker in tickers:
            logging.info("Getting prices for ticker " + ticker)
            data = rdp.HistoricalPricing.get_summaries(
                universe=ticker,
                interval= rdp.Intervals.DAILY,
                start="2018-07-01",
                #end="2021-01-01",
                count=365*3,
                fields=['OPEN_PRC', 'HIGH_1', 'LOW_1', 'TRDPRC_1', 'ACVOL_UNS']
            )
            if data.data.df is not None:
                logging.info("Upload Data for Ticker: " + ticker)
                for timestamp, row in data.data.df.iterrows():
                    date = str(timestamp.date())
                    open = row['OPEN_PRC']
                    high = row['HIGH_1']
                    low = row['LOW_1']
                    close = row['TRDPRC_1']
                    vol = row['ACVOL_UNS']
                    sql.uploadData(ticker, date, open, high, low, close, vol)

            else:
                logging.warning("No Historical Data for Ticker: " + ticker)

    def xxx(self, ticker):
        query = rdp.HistoricalPricing.get_summaries(ticker)
        if ticker in ["SHLG.DE", "IFXGn.DE"]:                   # For these Tickers there is no TRDPRC_1 instead
            print(query.data.df.columns.values)                 # there is OFF_CLOSE
        if "TRDPRC_1" in query.data.df.columns.values:
            pass
        else:
            print("TRDPRC_1 not available for Stock: " + ticker)

        """
        query = rdp.HistoricalPricing.get_summaries(
            universe=ticker,
            interval= rdp.Intervals.DAILY,
            start="2020-01-01",
            fields=['OPEN_PRC', 'HIGH_1', 'LOW_1', 'OFF_CLOSE', 'ACVOL_UNS']
        )
        display(query.data.df)
        """
        return query.data.df.columns.values