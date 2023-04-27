import click
import json


@click.command()
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to the json file")
@click.option("--number", required=True, type=int, help="Number of entries in resulting file")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--sort", is_flag=True, help="Sort the entries before cutting them")
def main(path: click.Path, number: int, output: click.Path, sort: bool):
    with open(str(path), "r") as config:
        list_fonts = json.load(config)
        assert type(list_fonts) == list, f"list_fonts is type {type(list_fonts)}"

    if sort:
        list_fonts = sorted(list_fonts)

    new_list_fonts = list_fonts[:number]
    with open(str(output), "w") as config:
        json.dump(new_list_fonts, config)


if __name__ == "__main__":
    main()
