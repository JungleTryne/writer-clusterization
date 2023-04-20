import click
import json


def cut_off_rule(font: str) -> bool:
    return int(font.split("/")[0]) <= 10000


@click.command()
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to the json file")
@click.option("--output", required=True, type=click.Path(), help="Output path")
def main(path: click.Path, output: click.Path):
    with open(str(path), "r") as config:
        list_fonts = json.load(config)
        assert type(list_fonts) == list, f"list_fonts is type {type(list_fonts)}"

    new_list_fonts = [font for font in list_fonts if cut_off_rule(font)]
    print(f"Purged: {len(list_fonts) - len(new_list_fonts)}")
    with open(str(output), "w") as config:
        json.dump(new_list_fonts, config)


if __name__ == "__main__":
    main()