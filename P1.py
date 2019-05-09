import math
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from moviepy.editor import VideoFileClip


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, belta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, belta, gamma)


def process_image(image):
	# NOTE: The output you return should be a color image (3 channel) for processing video below
	# TODO: put your pipeline here,
	# you should return the final output (image where lines are drawn on lanes)
    covered_area = np.array([[0, height], [int(width * 0.45), int(height * 0.6)], [
                            int(width * 0.55), int(height * 0.6)], [width, height]])

    img = grayscale(image)
    img1 = gaussian_blur(img, 5)
    img2 = canny(img1, 50, 150)
    img3 = hough_lines(img2, 1, 3.1416/180, 30, 30, 50)
    img4 = region_of_interest(img3, [covered_area])
    result = weighted_img(img4, image, α=0.8, β=1., γ=0.)

    return result


if __name__ == "__main__":
    # execute only if run as a script
    # os.listdir("test_images/")
    print(os.listdir("test_images/"))

    # TODO: Build your pipeline that will draw lane lines on the test_images
    # then save them to the test_images_output directory.
    items = os.listdir("test_images/")

    for item in items:
        print("Processing image ==>" + item)
        initial_img = mpimg.imread("test_images/" + item)
        img = grayscale(initial_img)
        (height, width) = img.shape
        img = gaussian_blur(img, 5)

        img2 = canny(img, 50, 150)
        img3 = hough_lines(img2, 1, 3.1416/180, 30, 30, 50)

        covered_area = np.array([[0, height], [int(width * 0.49), int(height * 0.6)], [
                                int(width * 0.51), int(height * 0.6)], [width, height]])
        img4 = region_of_interest(img3, [covered_area])

        img5 = weighted_img(img4, initial_img, α=0.8, β=1., γ=0.)

        plt.figure()
        plt.imshow(img5)

        # Import everything needed to edit/save/watch video clips

    # reading in an image
    image = mpimg.imread('test_images/solidYellowLeft.jpg')
    img7 = process_image(image)
    plt.imshow(img7)

    white_output = 'test_videos_output/challenge.mp4'
    # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    # To do so add .subclip(start_second,end_second) to the end of the line below
    # Where start_second and end_second are integer values representing the start and end of the subclip
    # You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    os.listdir("test_videos/")
    clip1 = VideoFileClip("test_videos/challenge.mp4")
    # NOTE: this function expects color images!!
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)
    # %time white_clip.write_videofile(white_output, audio=False)
