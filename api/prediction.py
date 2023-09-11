import cv2 as cv
import tensorflow as tf
import os
import numpy as np


def prediction(citra_tes, data_tes):
    img = cv.imread(citra_tes, cv.IMREAD_UNCHANGED)
    res = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
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

    model_load = tf.keras.models.load_model('./dataset/model/model.h5')
    model_load.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    hasil = model_load.predict(data_tes)
    hasil = np.argmax(hasil, axis=1)

    for i in cnts:
        for i in range(len(hasil)):
            if hasil[i] == 0:
                res = cv.drawContours(res, cnts[i], -1, (255, 0, 0), 6)
            elif hasil[i] == 1:
                res = cv.drawContours(res, cnts[i], -1, (0, 255, 0), 6)
            elif hasil[i] == 2:
                res = cv.drawContours(res, cnts[i], -1, (0, 0, 255), 6)

    current = os.getcwd()
    path = './dataset/result'
    os.chdir(path)
    base_filename = 'result'
    extension = '.jpg'
    counter = 1
    filename = f'{base_filename}-{counter}{extension}'

    while os.path.exists(filename):
        counter += 1
        filename = f'{base_filename}-{counter}{extension}'
    
    result = cv.imwrite(filename, res)
    os.chdir(current)

    total_pala = len(hasil)
    pala_a = (hasil == 0).sum()
    pala_b = (hasil == 1).sum()
    pala_c = (hasil == 2).sum()
    print(pala_a)
    print(pala_b)
    print(pala_c)
    print(total_pala)
    return total_pala, pala_a, pala_b, pala_c
