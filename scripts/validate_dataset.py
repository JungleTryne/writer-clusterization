import json

import click


@click.command()
@click.option("--train_json", required=True, type=str)
@click.option("--val_json", required=True, type=str)
@click.option("--test_json", required=True, type=str)
def main(train_json: str, val_json: str, test_json: str):
    fonts = []
    with open(train_json, "r") as file:
        fonts += json.load(file)

    with open(val_json, "r") as file:
        fonts += json.load(file)

    with open(test_json, "r") as file:
        fonts += json.load(file)

    fonts_numbers = sorted([int(font.split("/")[1].split("_")[0]) for font in fonts])
    assert len(fonts_numbers) == 10100, len(fonts_numbers)
    for i, font_id in enumerate(fonts_numbers):
        assert i == font_id

    print("OK")


if __name__ == "__main__":
    main()
