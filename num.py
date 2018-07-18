from imutils import contours
from PIL import Image
import pytesseract
import numpy as np
import argparse
import imutils
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-p", "--preprocess", type=str, default="blur", help="type of preprocessing to be done")
args = vars(ap.parse_args())

#kernel
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

#tratamento da imagem
image = cv2.imread(args["image"])
image = imutils.resize(image, width=700)
copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the output images
#cv2.imshow("Image", image1)
#cv2.imshow("gray tesseract", gray1)

# compute the Scharr gradient
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0,
    ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
#cv2.imshow("gradX", gradX)

# apply a closing operation using the rectangular kernel
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 200, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# apply a second closing operation to the binary image,
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
#cv2.imshow("thresh", thresh)


#achando os contornos
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#mostrando os retangulos e guardando as localizações
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if ar >3.5 and ar < 6.5:
        if (w > 0 and w < 120) and (h > 10 and h < 20):
            group = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            group = imutils.resize(group, width=400)

            # TESSERACT
            image1 = imutils.resize(image, width=400)
            # check to see if we should apply thresholding to preprocess the
            # image
            if args["preprocess"] == "thresh":
                group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # make a check to see if median blurring should be done to remove
            # noise
            elif args["preprocess"] == "blur":
                group = cv2.medianBlur(group, 3)

            # write the grayscale image to disk as a temporary file so we can
            # apply OCR to it
            filename = "{}.png".format(os.getpid())
            cv2.imwrite(filename, group)

            # load the image as a PIL/Pillow image, apply OCR, and then delete
            # the temporary file
            text = pytesseract.image_to_string(Image.open(filename))
            os.remove(filename)
            print(text)
            # cv2.imshow("group", group)
            # cv2.waitKey(0)

            # draw the digit classifications around the group
            cv2.rectangle(copy, (x - 5, y - 5),
                          (x + w + 5, y + h + 5), (0, 0, 255), 2)
            cv2.putText(copy,"".join(text), (x - 90, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if ar > 1.5 and ar < 3:
        if (w > 30 and w < 40) and (h > 10 and h < 20):
            group = gray[y - 5:y+ h + 5, x - 5:x + w + 5]
            group = imutils.resize(group, width=200)

            # TESSERACT
            image1 = imutils.resize(image, width=200)
            # check to see if we should apply thresholding to preprocess the
            # image
            if args["preprocess"] == "thresh":
                group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # make a check to see if median blurring should be done to remove
            # noise
            elif args["preprocess"] == "blur":
                group = cv2.medianBlur(group, 3)

            # write the grayscale image to disk as a temporary file so we can
            # apply OCR to it
            filename = "{}.png".format(os.getpid())
            cv2.imwrite(filename, group)

            # load the image as a PIL/Pillow image, apply OCR, and then delete
            # the temporary file
            text = pytesseract.image_to_string(Image.open(filename))
            os.remove(filename)
            print(text)
            # cv2.imshow("group", group)
            # cv2.waitKey(0)

            # draw the digit classifications around the group
            cv2.rectangle(copy, (x - 5, y - 5),
                          (x + w + 5, y + h + 5), (0, 255, 0), 2)
            cv2.putText(copy, "".join(text), (x + 50, y ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

cv2.imshow("Image",copy)


cv2.waitKey(0)