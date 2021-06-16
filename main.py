import numpy as np
import cv2


def line_detection(image, threshold):
    # Horizontal line detection mask
    mask_hori = np.array([
        [-1, -1, -1],
        [2, 2, 2],
        [-1, -1, -1]])
    # Vertical line detection mask
    mask_vert = np.array([
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1]])
    # plus 45 degrees line detection mask
    mask_p45 = np.array([
        [-1, -1, 2],
        [-1, 2, -1],
        [2, -1, -1]])
    # minus 45 degrees line detection mask
    mask_m45 = np.array([
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2]])

    m, n = image.shape # gets the image dimensions
    R = np.zeros((m, n)) # allocates space for the new response image

    for i in range(1, m - 1):  # loops through image x-axis position(i, j) is the middle element of the mask
        for j in range(1, n - 1):
            img=image[i - 1:i + 2, j - 1:j + 2] # saves the values of the respective pixels underneath the mask

            temp1 = mask_hori * img  # multiply the mask values by their respective pixel values
            temp2 = mask_vert * img
            temp3 = mask_p45 * img
            temp4 = mask_m45 * img

            # Response is given by the sum of all the weighted pixel responses
            R[i, j] = temp1.sum()+temp2.sum()+temp3.sum()+temp4.sum()

    for i in range(m):  # applies the pixel response threshold to the resulting image.
        for j in range(n):
            if R[i, j] > threshold:  # if the response is above the threshold then the value is changed to 255
                R[i, j] = 255
            else: # if the response is below the threshold then the value is changed to 0
                R[i, j] = 0

    return R


if __name__ == '__main__':

    image = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test7.jpg', 0) # loads the image using opencv

    image = cv2.resize(image, (720, 720))  # resizes the image using opencv to reduce processing
    cv2.imshow('Original Test Image', image)  # shows the original image post processing
    cv2.imwrite('Original Test Image3.jpg', image)  # writes to file the original image post processing

    threshold = 127  # sets the pixel response threshold
    detected_image = line_detection(image, threshold)  # function call

    cv2.imshow('Line detection result', detected_image) # shows the result of the line detection
    cv2.imwrite('Line detection result test3.jpg', detected_image) # writes to file the result of the line detection

    if cv2.waitKey(0) & 0xFF == 27:  # shows the images until a keyboard input is given
        cv2.destroyAllWindows()