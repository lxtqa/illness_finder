# below three lines bans GPU uses CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import os
from keras import optimizers
from keras import utils as np_utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import adagrad_v2
# 回调函数，每个训练批次调用一次
from keras.callbacks import ModelCheckpoint


# 动物数据预处理
imgdata_dir = 'sectionimages'

# 不使用数据增强
img_datagen = ImageDataGenerator(rescale=1./255)

mbatch_size = 20

# 使用数据增强
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# val_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# 使用迭代器生成图片张量
img_generator = img_datagen.flow_from_directory(imgdata_dir, target_size=(320, 320), batch_size=mbatch_size, class_mode='binary')
# 获取照片数量
img_count = img_generator.n
# print(img_generator.n)
# print(len(img_generator.labels))
print(img_count / 5)
img_cut = int(img_count / 5)
print(img_cut)
# 提取数据，因为构造器生成的数据标签是一维向量我们要分类10种不同的类型，所以需要将数据提取出来，并将标签one-hot
labels = []
datas = np.zeros((img_count, 320, 320 ,3))

for i in range(len(img_generator)):
    aa = img_generator.next()
    labels = np.hstack((labels, aa[1]))
    for j in range(len(aa[1])):
        datas[mbatch_size * i + j] = aa[0][j]

train_datas = datas[img_cut:]
train_labels = labels[img_cut:]

val_datas = datas[:img_cut]
val_labels = labels[:img_cut]

# print(resnet_base.summary())
# model = models.Sequential()


'''
resnet_base.trainable = False
flag = False
for layer in resnet_base.layers:
    if layer.name == 'res5c_branch2a':
        flag = True
    if flag:
        layer.trainable = True
'''
'''
setup_to_transfer_learning(model,base_model)
history_tl = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
model.save('E:/KaggleDatas/idenprof-jpg/idenprof/flowers17_iv3_tl.h5')
setup_to_fine_tune(model,base_model)
history_ft = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
model.save('E:/KaggleDatas/idenprof-jpg/idenprof/flowers17_iv3_ft.h5')
'''
'''
setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    epochs=5,
                    validation_data=val_generator,
                    validation_steps=5,
                    class_weight='auto'
                    )
model.save('E:/sectionimages/section-inceptionV3-more21.h5')
'''

# resnet_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# 构建基础模型
base_model = InceptionV3(weights='imagenet',include_top=False, input_shape=(320, 320, 3))

# 增加新的输出层
x = base_model.output
# GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = GlobalAveragePooling2D()(x) 
x = Dropout(rate=0.3)(x)
x = Dense(512,activation='relu')(x)
predictions = Dense(1,activation='sigmoid')(x)
model = Model(inputs=base_model.input,outputs=predictions)

'''
这里的base_model和model里面的iv3都指向同一个地址
'''
def setup_to_transfer_learning(model,base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 16 
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=adagrad_v2(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

setup_to_fine_tune(model,base_model)

# checkpoint
filepath="sectionimages/section-inceptionV3-more0323-1-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

history_ft = model.fit(x=train_datas, y=train_labels, batch_size=5, epochs=20, validation_data=(val_datas, val_labels), callbacks=callbacks_list)
'''
history_ft = model.fit_generator(generator=train_generator,
                                 epochs=10,
                                 validation_data=val_generator,
                                 validation_steps=5,
                                 callbacks=callbacks_list,
                                 class_weight='auto')
                                 '''
# model.save('E:/fireimages/fire-inceptionV3-more0318-0.h5')

'''
# 绘制训练精度损失曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='training acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('training & val accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('training & val loss')
plt.legend()

plt.show()
'''