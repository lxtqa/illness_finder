import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
 
from tensorflow import keras
from keras import models
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import os
import tkinter as tk
from tkinter import filedialog
 
model = models.load_model('sectionimages/section-inceptionV3-more21.h5')
 
root = tk.Tk()
root.withdraw()
 
for i in range(1,9):
    path = "sectionimages/test/"+str(i)+".jpg"
    img = image.load_img(path)
    img = img.resize((300, 300))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    j = 0
    for data in y[0]:
        if data >= 0.8:#可以调整阈值使得有病判断增大，data的值越大，有病的概率越高
            print(str(i)+'    illed {:.2f}'.format(data))
        else:
            print(str(i)+'    healthy {:.2f}'.format(data))
    

