import yaml

with open("Configuration/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

def check_prices(stock_name):
