import random

import PIL


ACTIONS = {
    0: {
        "name": "straight",
        "motors": (50, 50),
        "time": None
    },
    1: {
        "name": "up-left",
        "motors": (0, 35),
        "time": 400
    },
    2: {
        "name": "left",
        "motors": (0, 35),
        "time": 600
    },
    3: {
        "name": "up-right",
        "motors": (35, 0),
        "time": 400
    },
    4: {
        "name": "right",
        "motors": (35, 0),
        "time": 600
    },
    5: {
        "name": "na",
        "motors": (None, None),
        "time": 700
    }
}


def generate_label(ball_x: int, ball_y: int, image_width: int, image_height: int, image_horizon: int):
    half_vert_court = (image_height + image_horizon) / 2
    one_third_horiz = image_width / 3
    two_third_horiz = (2 * image_width) / 3

    if ball_x == 0 and ball_y == 0:
        return 5

    elif ball_x < one_third_horiz and ball_y > half_vert_court:
        return 1

    elif ball_x < one_third_horiz and ball_y < half_vert_court:
        return 2

    elif one_third_horiz <= ball_x <= two_third_horiz:
        return 0

    elif ball_x > two_third_horiz and ball_y > half_vert_court:
        return 3

    elif ball_x > two_third_horiz and ball_y < half_vert_court:
        return 4


def generate_dataset(dataset_size: int=10000):
    data = []
    labels = []
    ball = PIL.Image.open("ball.png")
    landscape = PIL.Image.open("landscape.png")
    landscape_height = landscape.size[1]
    landscape_width = landscape.size[0]
    landscape_horizon = (2 * landscape_height / 3)

    for i in range(dataset_size):
        if i%1000 == 1:
            print("Executed " + str(i) + " iterations")
        input_data = landscape.copy()

        if random.randint(1,4) == 1:
            ball_x = 0
            ball_y = 0

        else:
            offset = random.randint(-int(landscape_height / 20), int(landscape_height / 3)) - ball.size[1] / 2

            ball_x = random.randint(0, landscape_width)
            ball_y = int(landscape_horizon + offset)

            input_data.paste(ball, (ball_x, ball_y))
            # input_data.show()
            # input()
        data.append(input_data)
        labels.append(generate_label(ball_x, ball_y, landscape_width, landscape_height, landscape_horizon))

    return data, labels
