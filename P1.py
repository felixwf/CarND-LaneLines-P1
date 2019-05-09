import math
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from moviepy.editor import VideoFileClip

# Convert image into a gray one


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Detect the edge


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Blur the image


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Only use the interest part on the image


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

# Draw lines on a image, with red color


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Get Hough lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Add one image on another


def weighted_img(img, initial_img, alpha=0.8, belta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, belta, gamma)

# Pipeline to process the image, used for video stream


def process_image(initial_img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Step 1: Gray
    img = grayscale(initial_img)
    (img_h, img_w) = img.shape

    # Step 2: Blur
    blur_size = round(0.015 * min(img_h, img_w))
    if(blur_size % 2 != 1):
        blur_size = blur_size + 1
    # print(blur_size)
    img = gaussian_blur(img, blur_size)

    # Step 3: Canny
    img2 = canny(img, 50, 150)

    # Step 4: Hough
    img3 = hough_lines(img2, 1, 3.1416/180, 30, 30, 50)

    covered_area = np.array([[0, img_h], [int(img_w * 0.47), int(img_h * 0.6)], [
                            int(img_w * 0.53), int(img_h * 0.6)], [img_w, img_h]])
    
    # Step 5: Cover
    img4 = region_of_interest(img3, [covered_area])

    # Step 6: Add on the original image
    result = weighted_img(img4, initial_img, alpha=0.8, belta=1., gamma=0.)
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(initial_img)
    plt.subplot(2, 3, 2)
    plt.imshow(img)
    plt.subplot(2, 3, 3)
    plt.imshow(img2)
    plt.subplot(2, 3, 4)
    plt.imshow(img3)
    plt.subplot(2, 3, 5)
    plt.imshow(img4)
    plt.subplot(2, 3, 6)
    plt.imshow(result)

    return result


# Main progress
if __name__ == "__main__":

    # TODO: Build your pipeline that will draw lane lines on the test_images
    # then save them to the test_images_output directory.
    items = os.listdir("test_images/")
    # plt.figure()
    index = 1
    for item in items:
        print("Processing image ==>" + item)
        initial_img = mpimg.imread("test_images/" + item)


        # # Step 1: Gray
        # img = grayscale(initial_img)
        # (img_h, img_w) = img.shape

        # # Step 2: Blur
        # blur_size = round(0.015 * min(img_h, img_w))
        # if(blur_size % 2 != 1):
        #     blur_size = blur_size + 1
        # # print(blur_size)
        # img = gaussian_blur(img, blur_size)

        # # Step 3: Canny
        # img2 = canny(img, 50, 150)

        # # Step 4: Hough
        # img3 = hough_lines(img2, 1, 3.1416/180, 30, 30, 50)

        # covered_area = np.array([[0, img_h], [int(img_w * 0.49), int(img_h * 0.6)], [
        #                         int(img_w * 0.51), int(img_h * 0.6)], [img_w, img_h]])
        
        # # Step 5: Cover
        # img4 = region_of_interest(img3, [covered_area])

        # # Step 6: Add on the original image
        # img5 = weighted_img(img4, initial_img, alpha=0.8, belta=1., gamma=0.)
        # plt.figure(1)
        # plt.subplot(2, 3, index)
        # plt.imshow(img)
        # plt.figure(2)
        # plt.subplot(2, 3, index)
        # plt.imshow(img2)
        # plt.figure(3)
        # plt.subplot(2, 3, index)
        # plt.imshow(img3)
        # plt.figure(4)
        # plt.subplot(2, 3, index)
        # plt.imshow(img4)
        # plt.figure(5)
        # plt.subplot(2, 3, index)
        # plt.imshow(img5)

        # plt.figure(6)
        # plt.subplot(2, 3, index)
        plt.imshow(process_image(initial_img))
        # index = index + 1
    plt.show()
        # Import everything needed to edit/save/watch video clips

    # # reading in an image
    # image = mpimg.imread('test_images/solidYellowLeft.jpg')
    # img7 = process_image(image)
    # plt.imshow(img7)

    # white_output = 'test_videos_output/challenge.mp4'
    # # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    # # To do so add .subclip(start_second,end_second) to the end of the line below
    # # Where start_second and end_second are integer values representing the start and end of the subclip
    # # You may also uncomment the following line for a subclip of the first 5 seconds
    # # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    # os.listdir("test_videos/")
    # clip1 = VideoFileClip("test_videos/challenge.mp4")
    # # NOTE: this function expects color images!!
    # white_clip = clip1.fl_image(process_image)
    # white_clip.write_videofile(white_output, audio=False)
    # # %time white_clip.write_videofile(white_output, audio=False)
