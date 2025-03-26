import os
import sys
import importlib
import refinitiv.dataplatform as rdp
import yaml
import json
from IPython.display import display


def show_help():
    print("Usage: vc-analysis [subcommand] [options]")
    print("Available subcommands:")
    print("  help         Show this help message")
    print("  charts       Generate and analyze charts")
    print("Use 'vc-analysis [subcommand] help' for details on a specific command.")

def load_config(file_path="Configuration/config.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def main():
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)

    subcommand = sys.argv[1]
    args = sys.argv[2:]

    if subcommand == "help":
        show_help()
        sys.exit(0)

    try:
        module = importlib.import_module(f"subcommands.{subcommand}")
        module.run(args)
    except ModuleNotFoundError:
        print(f"Error: Unknown subcommand '{subcommand}'")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    config = load_config()
    rdp.open_platform_session(
        config.get("refinitiv", {}).get("APP-ID"),
        rdp.GrantPassword(
            config.get("refinitiv", {}).get("USER"),
            config.get("refinitiv", {}).get("PASSWORD")
        )
    )
    test = rdp.HistoricalPricing.get_summaries('VOD.L',
                                               interval = rdp.Intervals.DAILY,
                                               start="2025-01-01",
                                               end="2025-02-01",
                                               fields = ['MKT_OPEN','MKT_HIGH', 'MKT_LOW', 'HIGH_1', 'TRDPRC_1', 'TRNOVR_UNS'])
    #display(test.data.df)
    #print(list(test.data.df.columns.values))
    #print(test.data.df.index)
    print(test.data.df.iloc[0].name.day)
    print(test.data.df.iloc[0].name.month)
    print(test.data.df.iloc[0].name.year)
    print(type(test.data.df.iloc[0].name))
    #print(test.data.df.to_dict())))
    #print(yaml.dump(test.data.df.to_dict(), default_flow_style=False))
    rdp.close_session()
    #main()
