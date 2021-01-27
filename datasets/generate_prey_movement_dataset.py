import random


ACTIONS = {
    0: {
        "name": "up",
        "motors": (50, 50),
        "time": None
    },
    1: {
        "name": "up-left",
        "motors": (0, 35),
        "time": 700
    },
    2: {
        "name": "left",
        "motors": (0,50),
        "time": 1000
    },
    3: {
        "name": "up-right",
        "motors": (35, 0),
        "time": 700
    },
    4: {
        "name": "right",
        "motors": (50, 0),
        "time": 1000
    },
    5: {
        "name": "back",
        "motors": (-30, -30),
        "time": 700
    }
}


def generate_eversive_move(values: list) -> int:
    back_right, back_center, back_left, right_ir, half_right_ir, center_ir, half_left_ir , left_ir = values

    if all(vals == 0 for vals in values):
        return 0

    elif center_ir < -0.3:
        return 5

    elif all(half_left_ir <= vals for vals in values) or half_left_ir < -0.25 or \
            all(back_left <= vals for vals in values) or back_left < -0.25:
        return 4

    elif all(left_ir <= vals  for vals in values) or left_ir < -0.25:
        return 3

    elif all(half_right_ir <= vals for vals in values) or half_right_ir < -0.25 or \
        all(back_right <= vals for vals in values) or back_right < -0.25:
        return 2

    elif all(right_ir <= vals for vals in values) or right_ir < -0.25:
        return 1

    elif all(center_ir <= vals for vals in values) or all(back_center <= vals for vals in values) or back_center < -0.25:
        return 0


def generate_dataset(dim: int=100000):
    data = []
    labels = []
    for _ in range(dim):
        if random.randint(1,8) == 1:
            input_data = [0,0,0,0,0,0,0,0]
        else:
            choice = random.randint(1,5)
            if choice == 1:
                input_data = [
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                    random.uniform(-0.45, -0.2),
                              random.uniform(-0.35, -0.1),
                              random.uniform(-0.2, 0),
                              random.uniform(-0.1, 0),
                              0]
            elif choice == 2:
                input_data = [
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                    random.uniform(-0.35, -0.1),
                              random.uniform(-0.45, -0.2),
                              random.uniform(-0.35, -0.1),
                              random.uniform(-0.2, 0),
                              random.uniform(-0.1, 0)]
            elif choice == 3:
                input_data = [
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                    random.uniform(-0.2, 0),
                              random.uniform(-0.35, -0.1),
                              random.uniform(-0.45, -0.2),
                              random.uniform(-0.35, -0.1),
                              random.uniform(-0.2, 0)]
            elif choice == 4:
                input_data = [
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                              random.uniform(-0.2, 0),
                              random.uniform(-0.35, -0.1),
                              random.uniform(-0.45, -0.2),
                              random.uniform(-0.35, -0.1)]
            elif choice == 5:
                input_data = [
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            random.uniform(-0.1, 0),
                            0,
                            random.uniform(-0.1, 0),
                            random.uniform(-0.2, 0),
                            random.uniform(-0.35, -0.1),
                            random.uniform(-0.45, -0.2)]
            elif choice == 6:
                input_data = [
                    random.uniform(-0.45, -0.2),
                    random.uniform(-0.35, -0.1),
                    random.uniform(-0.2, 0),
                    0,
                    0,
                    0,
                    0,
                    0
                ]
            elif choice == 7:
                input_data = [
                     random.uniform(-0.35, -0.1),
                     random.uniform(-0.45, -0.2),
                     random.uniform(-0.35, -0.1),
                    0,
                    0,
                    0,
                    0,
                    0
                ]
            else:

                input_data = [
                    random.uniform(-0.2, 0),
                    random.uniform(-0.35, -0.1),
                    random.uniform(-0.45, -0.2),
                    0,
                    0,
                    0,
                    0,
                    0
                ]
        assert len(input_data) == 8
        data.append(input_data)
        labels.append(int(generate_eversive_move(input_data)))

    return data, labels
