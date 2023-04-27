import click
import json


@click.command()
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to the json file")
def main(path: click.Path):
    with open(str(path), "r") as config:
        list_fonts = json.load(config)
        assert type(list_fonts) == list, f"list_fonts is type {type(list_fonts)}"

    print("Number of entries:", len(list_fonts))


if __name__ == "__main__":
    main()
