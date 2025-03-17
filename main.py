import sys
import importlib


def show_help():
    print("Usage: vc-analysis [subcommand] [options]")
    print("Available subcommands:")
    print("  help         Show this help message")
    print("  charts       Generate and analyze charts")
    print("Use 'vc-analysis [subcommand] help' for details on a specific command.")


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
    main()
