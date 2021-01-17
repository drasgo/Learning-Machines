import random

import PIL


def generate_label(ball_x, ball_y):
    pass


def generate_dataset(dim: int=10000):
    data = []
    labels = []
    ball = PIL.Image.open("ball.png")
    landscape = PIL.Image.open("tt.png")
    image_height = landscape.size[1]
    image_width = landscape.size[0]

    for _ in range(dim):

        if random.randint(1,4) == 1:
            input_data = landscape
            ball_x = 0
            ball_y = 0

        else:
            input_data = landscape.copy()
            offset = random.randint(-int(image_height / 20), int(image_height / 3)) - ball.size[1] / 2

            ball_x = random.randint(0, image_width)
            ball_y = int((2 * image_height / 3) + offset)

            input_data.paste(ball, (ball_x, ball_y))

        data.append(input_data)
        labels.append(generate_label(ball_x, ball_y))
