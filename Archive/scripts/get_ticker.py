import refinitiv.dataplatform as rdp
import progressbar
import pandas as pd
import yaml
import logging

def load_config(file_path="Configuration/config.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def open_rdp(config):
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

def get_cusip():
    # YAML-Datei einlesen
    with open("Configuration/tickers.yaml", "r") as file:
        data = yaml.safe_load(file)

    # Die Liste aus dem YAML-Dictionary extrahieren
    input_CUSIP = data.get("tickers", [])
    return input_CUSIP

def get_tickers(cusips):
    output = []

    bar = progressbar.ProgressBar(widgets=[
        'Get Tickers: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()
    ])
    for cusip in bar(cusips):
        try:
            df = rdp.search(
                view=rdp.SearchViews.EquityQuotes,
                query=cusip
            )
            output.append((cusip, df['RIC'][0], df['DocumentTitle'][0]))
        except:
            pass
    return output

def main():
    logging.info("Load Configuration")
    config = load_config()
    logging.info("Open Refinitiv Session")
    open_rdp(config)
    logging.info("Get CUSIPs from YAML")
    cusips = get_cusip()
    logging.info("Get Tickers from Refinitiv Session")
    tickers = get_tickers(cusips)
    logging.info("Export to Excel")
    # DataFrame erstellen
    df = pd.DataFrame(tickers, columns=["USIP", "Ticker", "Name"])
    # In eine Excel-Datei schreiben
    df.to_excel("ausgabe.xlsx", index=False)
    print("Von " + str(len(cusips)) + " CUSIP, wurden " + str(len(tickers)) + " gefunden!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="../runtime.log", filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    main()