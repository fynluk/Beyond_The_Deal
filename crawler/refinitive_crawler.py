import refinitiv.dataplatform as rdp
import logging
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
            logging.error("Ticker may be wrong: " + ticker)
        if name is None:
            logging.error("Name from ticker " + ticker + " not found")
            return "N/A"
        else:
            return name


    def testus(self):
        test2 = rdp.HistoricalPricing.get_summaries('VOD.L',
                                                   interval=rdp.Intervals.DAILY,
                                                   start="2025-01-01",
                                                   end="2025-02-01",
                                                   fields=['MKT_OPEN', 'MKT_HIGH', 'MKT_LOW', 'HIGH_1', 'TRDPRC_1',
                                                           'TRNOVR_UNS'])
        return test2

    def init(self, deals):
        pass