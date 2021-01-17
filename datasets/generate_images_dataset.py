import random

import PIL


def generate_label(ball_x, ball_y, image_width, image_horizon):
    pass


def generate_dataset(dataset_size: int=10000):
    data = []
    labels = []
    ball = PIL.Image.open("ball.png")
    landscape = PIL.Image.open("landscape.png")
    landscape_height = landscape.size[1]
    landscape_width = landscape.size[0]
    landscape_horizon = (2 * landscape_height / 3)

    for _ in range(dataset_size):
        input_data = landscape.copy()

        if random.randint(1,4) == 1:
            ball_x = 0
            ball_y = 0

        else:
            offset = random.randint(-int(landscape_height / 20), int(landscape_height / 3)) - ball.size[1] / 2

            ball_x = random.randint(0, landscape_width)
            ball_y = int(landscape_horizon + offset)

            input_data.paste(ball, (ball_x, ball_y))

        data.append(input_data)
        labels.append(generate_label(ball_x, ball_y, landscape_width, landscape_horizon))

    return data, labels
