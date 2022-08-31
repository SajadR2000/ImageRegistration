import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from q3v2 import registering, cropping, axis_interval_after_shift
from time import time


def running(img_file_name):
    img = cv.imread(img_file_name, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Image wasn't loaded!!")

    # Separating different channels
    height = img.shape[0] // 3
    B_channel = img[:height, :]
    G_channel = img[height:2 * height, :]
    R_channel = img[2 * height:3 * height, :]
    # Storing original image for future plotting
    unregistered = np.zeros((B_channel.shape[0], B_channel.shape[1], 3))
    unregistered[:, :, 0] = R_channel
    unregistered[:, :, 1] = G_channel
    unregistered[:, :, 2] = B_channel
    unregistered = unregistered.astype(np.uint8)
    # Finding optimal shift parameters.
    # Fixed channel: Blue channel
    # Moving channel 1: Green channel
    # Moving channel 2: Red channel
    # x-axis = horizontal direction
    # y-axis = vertical direction

    dx_moving1, dy_moving1, shift_x_list1, shift_y_list1 = registering(B_channel, G_channel)
    dx_moving2, dy_moving2, shift_x_list2, shift_y_list2 = registering(B_channel, R_channel)

    # To debug the cropping process you can use these optimal shift values
    # # Amir
    # if img_file_name == "Amir.tif":
    #     dx_moving1, dy_moving1 = 24, 49
    #     dx_moving2, dy_moving2 = 43, 103
    # # Mosque
    # elif img_file_name == "Mosque.tif":
    #     dx_moving1, dy_moving1 = -3, 55
    #     dx_moving2, dy_moving2 = -1, 124
    # # Train
    # elif img_file_name == "Train.tif":
    #     dx_moving1, dy_moving1 = 5, 43
    #     dx_moving2, dy_moving2 = 31, 87

    print("Fixing the blue channel the optimal shift values for " + img_file_name + " are:")
    print("Green channel: dx=", dx_moving1, " dy=", dy_moving1)
    print("Red channel: dx=", dx_moving2, " dy=", dy_moving2)
    # Shifting and finding the valid area. Explained in the report.
    x_interval_fixed, x_interval_moving1, x_interval_moving2 = axis_interval_after_shift(dx_moving1,
                                                                                         dx_moving2,
                                                                                         B_channel.shape,
                                                                                         1)
    y_interval_fixed, y_interval_moving1, y_interval_moving2 = axis_interval_after_shift(dy_moving1,
                                                                                         dy_moving2,
                                                                                         B_channel.shape,
                                                                                         0)

    B_channel_shifted = B_channel[y_interval_fixed[0]:y_interval_fixed[1], x_interval_fixed[0]:x_interval_fixed[1]]
    G_channel_shifted = G_channel[y_interval_moving1[0]:y_interval_moving1[1], x_interval_moving1[0]:x_interval_moving1[1]]
    R_channel_shifted = R_channel[y_interval_moving2[0]:y_interval_moving2[1], x_interval_moving2[0]:x_interval_moving2[1]]

    # Cropping the image. 45 is the maximum cropping percentage
    B_channel_cropped, G_channel_cropped, R_channel_cropped = cropping(B_channel_shifted,
                                                                       G_channel_shifted,
                                                                       R_channel_shifted,
                                                                       45)
    # Storing shifted image for future plotting
    colored_shifted = np.zeros((B_channel_shifted.shape[0], B_channel_shifted.shape[1], 3))
    colored_shifted[:, :, 0] = R_channel_shifted
    colored_shifted[:, :, 1] = G_channel_shifted
    colored_shifted[:, :, 2] = B_channel_shifted
    colored_shifted = colored_shifted.astype(np.uint8)
    # Storing cropped image
    colored_cropped = np.zeros((B_channel_cropped.shape[0], B_channel_cropped.shape[1], 3))
    colored_cropped[:, :, 0] = R_channel_cropped
    colored_cropped[:, :, 1] = G_channel_cropped
    colored_cropped[:, :, 2] = B_channel_cropped
    colored_cropped = colored_cropped.astype(np.uint8)

    # To see the intermediate steps, uncomment the following lines
    # for i in range(len(shift_x_list1)):
    #     plt.figure()
    #     # Sampling rate
    #     d = 2 ** (4 - i)
    #     # print(d)
    #     # Sampling the image
    #     B_channel_ = B_channel[::d, ::d]
    #     G_channel_ = G_channel[::d, ::d]
    #     R_channel_ = R_channel[::d, ::d]
    #     # print(B_channel_.shape)
    #     # print(G_channel_.shape)
    #     # print(R_channel_.shape)
    #     # print(shift_x_list1[i])
    #     # print(shift_y_list1[i])
    #     # print(shift_x_list2[i])
    #     # print(shift_y_list2[i])
    #     # Doing all steps mentioned above for the sampled image.
    #     x_interval_fixed, x_interval_moving1, x_interval_moving2 = axis_interval_after_shift(shift_x_list1[i],
    #                                                                                          shift_x_list2[i],
    #                                                                                          B_channel_.shape,
    #                                                                                          1)
    #     y_interval_fixed, y_interval_moving1, y_interval_moving2 = axis_interval_after_shift(shift_y_list1[i],
    #                                                                                          shift_y_list2[i],
    #                                                                                          B_channel_.shape,
    #                                                                                          0)
    #
    #     B_channel_shifted = B_channel_[y_interval_fixed[0]:y_interval_fixed[1], x_interval_fixed[0]:x_interval_fixed[1]]
    #     G_channel_shifted = G_channel_[y_interval_moving1[0]:y_interval_moving1[1],
    #                         x_interval_moving1[0]:x_interval_moving1[1]]
    #     R_channel_shifted = R_channel_[y_interval_moving2[0]:y_interval_moving2[1],
    #                         x_interval_moving2[0]:x_interval_moving2[1]]
    #
    #     I = np.zeros((B_channel_shifted.shape[0], B_channel_shifted.shape[1], 3))
    #     I[:, :, 0] = R_channel_shifted
    #     I[:, :, 1] = G_channel_shifted
    #     I[:, :, 2] = B_channel_shifted
    #     I = I.astype(np.uint8)
    #     # print(I.shape)
    #     # print("____________________________________________________________________")
    #     plt.imshow(I)

    plt.figure()
    plt.subplot(131)
    plt.imshow(unregistered)
    plt.title("Original Image")
    plt.subplot(132)
    plt.imshow(colored_shifted)
    plt.title("After Registering")
    plt.subplot(133)
    plt.imshow(colored_cropped)
    plt.title("After Cropping")

    return colored_cropped


t1 = time()
f_name = "Amir.tif"
res03 = running(f_name)
plt.imsave("res03-Amir.jpg", res03)

f_name = "Mosque.tif"
res04 = running(f_name)
plt.imsave("res04-Mosque.jpg", res04)

f_name = "Train.tif"
res05 = running(f_name)
plt.imsave("res05-Train.jpg", res05)

print("Processing each image took ", str((time()-t1) / 3), " seconds on average.")
#
# f_name = "test4.tif"
# res03 = running(f_name)
#
# f_name = "test1.tif"
# res04 = running(f_name)
#
# f_name = "test2.tif"
# res05 = running(f_name)
#
# f_name = "test3.tif"
# res03 = running(f_name)
#
# f_name = "three_generations.tif"
# res04 = running(f_name)
#
# f_name = "melons.tif"
# res05 = running(f_name)
#
# f_name = "girls.tif"
# res05 = running(f_name)
#
# f_name = "test5.tif"
# res05 = running(f_name)
#
# f_name = "test_4.tif"
# res03 = running(f_name)
#
# f_name = "test_1.tif"
# res04 = running(f_name)
#
# f_name = "test_2.tif"
# res05 = running(f_name)
#
# f_name = "test_3.tif"
# res03 = running(f_name)

plt.show()
