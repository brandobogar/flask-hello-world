import cv2 as cv
import numpy as np
import os



from get_latest import get_latest_uploads

def create_datates():
    img = get_latest_uploads()
    img = cv.imread(img, cv.IMREAD_UNCHANGED)
    res = cv.resize(img, (500, 500))
    # res = resize(image=img)
    contrast = cv.convertScaleAbs(res, beta=22.0, alpha=1.1)
    grey = cv.cvtColor(contrast, cv.COLOR_BGR2GRAY)
    mask = cv.threshold(grey, 128, 255, cv.THRESH_BINARY)[1]

    kernel = np.ones((10, 10), np.uint8)

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    mask = cv.GaussianBlur(mask, (0, 0), sigmaX=2,
                           sigmaY=2, borderType=cv.BORDER_DEFAULT)

    mask = (2*(mask.astype(np.float32))-255.0).clip(0, 255).astype(np.uint8)

    result = res.copy()
    result = cv.cvtColor(result, cv.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    outline = cv.Canny(mask, 30, 150)
    (cnts, _) = cv.findContours(outline, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(cnts, key=cv.contourArea, reverse=True)

    root= os.getcwd()
    save_crop_path = r'dataset/data_tes/data'
    for file in os.scandir(save_crop_path):
        os.remove(file.path)

    os.chdir(save_crop_path)

    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv.boundingRect(c)
        cropped_contour = res[y:y+h, x:x+w]
        image_name = "data_tes-" + str(i+1) + ".jpg"
        cv.imwrite(image_name, cropped_contour)
        readimage = cv.imread(image_name)
    # root = r'E:\belajar\react\palakowe\backend'
    
    os.chdir(root)

