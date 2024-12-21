import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.optimizers import SGD

# Встановлення фіксованого значення для генератора випадкових чисел
np.random.seed(42)

# Завантаження даних
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Параметри моделі
batch_size = 32
nb_classes = 10
nb_epoch = 25
img_rows, img_cols = 32, 32
img_channels = 3

# Нормалізація даних
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Перетворення міток у формат one-hot
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

# Створення моделі
model = Sequential()

# Додавання шарів згорткової нейромережі
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                 input_shape=(img_rows, img_cols, img_channels)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

# Компіляція моделі
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Навчання моделі
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, 
          validation_split=0.1, shuffle=True, verbose=2)

# Оцінка на тестових даних
scores = model.evaluate(X_test, Y_test, verbose=0)
print(f"Accuracy on test data: {scores[1] * 100:.2f}%")

# Збереження моделі
model.save('my_model.h5')
