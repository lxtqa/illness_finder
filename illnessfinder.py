from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import os
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.optimizers import adagrad_v2
 
 
# 数据预处理
train_path = 'sectionimages/train'
val_path = 'sectionimages/val'
 
# 不使用数据增强
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
 
# 使用数据增强
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# val_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
 
# 使用迭代器生成图片张量
train_generator = train_datagen.flow_from_directory(train_path, target_size=(300, 300), batch_size=10, class_mode='binary')
val_generator = train_datagen.flow_from_directory(val_path, target_size=(300, 300), batch_size=10, class_mode='binary')
 
# resnet_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# 构建基础模型
base_model = InceptionV3(weights='imagenet',include_top=False, input_shape=(300, 300, 3))
 
# 增加新的输出层
x = base_model.output
# GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = GlobalAveragePooling2D()(x) 
x = Dense(512,activation='relu')(x)
predictions = Dense(1,activation='sigmoid')(x)
model = Model(inputs=base_model.input,outputs=predictions)
 
'''
这里的base_model和model里面的iv3都指向同一个地址
'''

for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history_tl = model.fit_generator(generator=train_generator,
                    epochs=5,
                    validation_data=val_generator,
                    validation_steps=5,
                    )
model.save('sectionimages/section-inceptionV3-more21.h5')
