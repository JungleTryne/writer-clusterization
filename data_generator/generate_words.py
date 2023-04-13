from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 64


def generate_image(content: str, font_path: str, image_res: Tuple[int, int]) -> Image:
    image = Image.new("RGB", image_res, BACKGROUND_COLOR)
    font = ImageFont.truetype(font_path, FONT_SIZE)
    draw = ImageDraw.Draw(image)
    text_length = draw.textlength(content, font)
    draw.text(
        (image_res[0] // 2 - text_length // 2, 0),
        content,
        fill=TEXT_COLOR,
        font=font
    )
    return image


def main():
    image = generate_image("Incredible", "./fonts/font.ttf", (224, 112))
    image.save("output.png")


if __name__ == "__main__":
    main()