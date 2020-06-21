from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
#data set generator
train=image.ImageDataGenerator(rescale=1./255
                               ,
                               horizontal_flip=True
                               ,
                               shear_range=.1,
                               zoom_range=.1)
train_set=train.flow_from_directory('C:\\Users\\ahmed\\PycharmProjects\\untitled\\catanddog\\train\\',
                          target_size=(100,100),
                          batch_size=25,
                          class_mode='binary')
model=Sequential()
model.add(layers.Conv2D(200,(3,3),input_shape=(100,100,3),activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
model.add(layers.Conv2D(32,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(units=128,activation='relu'))
model.add(layers.Dense(units=128,activation='relu'))

model.add(layers.Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_set,epochs=200 , steps_per_epoch=10)
#1
for repeat in range(1,20):
    img1=image.load_img('C:\\Users\\ahmed\\PycharmProjects\\untitled\\catanddog\\test1\\{}.jpg'.format(repeat),target_size=(100, 100))
    img=image.img_to_array(img1)
    img = img/255
    img=np.expand_dims(img,axis=0)
    prediction=model.predict_classes(img)
    plt.text(20, 62, prediction, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
    plt.imshow(img1)
    plt.show()
#2

