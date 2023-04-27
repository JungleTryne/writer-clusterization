import click
import json


@click.command()
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to the json file")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--ratio", required=True, type=float)
def main(path: click.Path, output: click.Path, ratio: float):
    with open(str(path), "r") as config:
        list_fonts = json.load(config)
        assert type(list_fonts) == list, f"list_fonts is type {type(list_fonts)}"

    new_list_fonts = list_fonts[:int(len(list_fonts) * ratio)]
    with open(str(output), "w") as config:
        json.dump(new_list_fonts, config)


if __name__ == "__main__":
    main()