import numpy as np
from keras.models import load_model
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# Завантаження моделі
model = load_model('my_model.h5')

# Завантаження даних для перевірки
(_, _), (X_test, y_test) = cifar10.load_data()

# Нормалізація даних
X_test = X_test.astype('float32') / 255

# Словник класів CIFAR-10
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Вибір випадкового зображення для тесту
idx = np.random.randint(0, X_test.shape[0])
test_image = X_test[idx]
test_label = y_test[idx][0]

# Підготовка зображення для моделі
test_image_input = np.expand_dims(test_image, axis=0)

# Прогноз моделі
predictions = model.predict(test_image_input)
predicted_class = np.argmax(predictions)

# Вивід результатів
print(f"True class: {class_labels[test_label]}")
print(f"Predicted class: {class_labels[predicted_class]}")

# Вивід зображення
plt.imshow(test_image)
plt.title(f"True: {class_labels[test_label]}, 
          Predicted: {class_labels[predicted_class]}")
plt.axis('off')
plt.show()
