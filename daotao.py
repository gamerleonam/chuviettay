from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

#1.nap du lieu MNIST
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

#2.xay dung mo hinh CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#hien thi cau truc cua mo hinh
model.summary()

#3.chuan bi du lieu
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype(('float32')) / 255
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype(('float32')) / 255
train_labels = to_categorical(train_labels)
train_labels = to_categorical(test_images)

#4.bien dich mo hinh
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#5.huan luyen mo hinh#
modle.fit(train_images,train_labels,epochs =5,batvh_size=64)
#6.danh gia mo hinh kiem thu
test_loss,test_acc=model.evaluate(test_images,test_labels)
#in ra do do chinh xac
print(test_acc)
#luu mo hinh da huan luyen
model.save('mnist.h5')