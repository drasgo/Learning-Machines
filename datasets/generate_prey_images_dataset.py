import os
import shutil
import numpy as np
import PIL
from cv2 import cv2
from tqdm import tqdm

from utils import save_list_to_pickle_file


def generate_label(pixel_position):
    if pixel_position == -1:
        return 5
    elif pixel_position < 0.33:
        return 2
    elif 0.33 <= pixel_position < 0.66:
        return 0
    elif pixel_position > 0.66:
        return 4


def check_prey_in_image(im_list):
    horizontal_position = -1
    # Iterate through rows (from the bottom to  the topof the image,  it should recognize closer targets)
    for row_index in range(len(im_list) - 1, -1, -1):

        # Iterate through columns (meaning that each "column index" is the index of a pixel)
        for column_index in range(len(im_list[row_index])):
            # Pixel cell with r g b values
            pixel = im_list[row_index][column_index]
            # If the pixel is red look in the same row if there are other green pixels. If so, check what is the last
            # red pixel and save it. So, the average red pixel position on the horizontal axis is the average of these two
            if pixel[0] > 175 and pixel[1] < 40 and pixel[2] < 40:
                last_pixel = -1

                for other_pixels in range(column_index + 1, len(im_list[row_index])):
                    other_pixel = im_list[row_index][other_pixels]
                    if other_pixel[0] > 175 and other_pixel[1] < 40 and other_pixel[2] < 40:
                        continue
                    last_pixel = other_pixels - 1
                    break

                if last_pixel == -1:
                    average_pixel = column_index

                else:
                    average_pixel = (last_pixel + column_index) / 2

                # the horizontal position is a perccentage between 0 (far left) and 1 (far right). This will be
                # divided in 5 slots which will be used for finding the labels of the 5 images.
                # If no red pixel was found, than the horiizontal_axis is -1 and the label will be "no target in the image"
                horizontal_position = average_pixel / len(im_list[row_index])
                break
        if horizontal_position != -1:
            break

    if horizontal_position != -1:
        flag = True
    else:
        flag = False

    return horizontal_position


def generate_dataset(folder="images/"):
    data = []
    labels = []
    images_names = []
    index = 0
    # Walks through all images
    for _, _, names in os.walk(folder):
        # Iterates through every image in the folder
        for counter, name in tqdm(enumerate(names)):
            # Save partial images-labels results
            if counter % 1000 == 0:
                index += 1
                save_list_to_pickle_file(data, "data" + str(index))
                save_list_to_pickle_file(labels, "labels" + str(index))

                # Clean up data and labels
                # remove image files
                for im_name in images_names:
                    os.remove(folder + im_name)
                images_names = []

            # Retrieve image and transform it to list
            im = cv2.imread(folder + name, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).tolist()
            image = PIL.Image.fromarray(np.array(im, dtype=np.uint8))
            image = image.convert("RGB")
            label = generate_label(check_prey_in_image(im))

            data.append(image)
            labels.append(label)
            images_names.append(name)
            del im
            del image

    return data, labels

def purge_dataset(folder="images/"):
    name_empty_images = []
    name_prey_images = []
    # print("here")
    for _, _, files in os.walk(folder):
        for idx, name in enumerate(files):
            if idx % 2000 == 1999:
                print("done " + str(idx) + " so far. Partial results: " + str(len(name_prey_images)) +
                       " images with preys and " + str(len(name_empty_images)) + " without preys.")
            im = cv2.imread(folder + name, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).tolist()
            # image = PIL.Image.fromarray(np.array(im, dtype=np.uint8))
            # image = image.convert("RGB")
            if check_prey_in_image(im) == -1:
                name_prey_images.append(name)
            else:
                name_empty_images.append(name)
            # image.show()
            # input()
    print("number of images with prey in it: " + str(len(name_prey_images)))
    print("number of images without prey in it: " + str(len(name_empty_images)))
    input()
    while len(name_prey_images) < len(name_empty_images):
        try:
            os.remove(folder + name_empty_images.pop(0))
        except OSError:
            pass
