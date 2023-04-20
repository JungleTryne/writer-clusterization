import json
import os

import click
import yaml
from tqdm import tqdm


def fix(root, fonts_list, words_list):
    new_fonts = []
    for font in tqdm(fonts_list):
        broken = False
        for word in words_list:
            path = os.path.join(root, font, f"{word}.jpg")
            if not os.path.exists(path):
                broken = True
                break
        if not broken:
            new_fonts.append(font)

    print(f"Purged: {len(fonts_list) - len(new_fonts)}")
    return new_fonts


def fix_file(root, config, font_key, word_key):
    with open(os.path.join(root, config["dataset"][font_key])) as file:
        fonts_list = json.load(file)
    with open(os.path.join(root, config["dataset"][word_key])) as file:
        words_list = json.load(file)
    new_fonts = fix(root, fonts_list, words_list)
    with open(os.path.join(root, config["dataset"][font_key]), "w") as file:
        json.dump(new_fonts, file)


@click.command()
@click.option("--config_path", required=True, type=str)
def main(config_path: str):
    with open(str(config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    root = config["dataset"]["root_path"]
    fix_file(root, config, "fonts_train", "words_train")
    fix_file(root, config, "fonts_test", "words_test")
    fix_file(root, config, "fonts_val", "words_val")


if __name__ == "__main__":
    main()
