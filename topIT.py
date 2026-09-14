# Импорт библиотеки TensorFlow для работы с нейронными сетями
import tensorflow as tf
# Импорт TensorFlow Hub для использования предобученных моделей
import tensorflow_hub as hub
# Импорт NumPy для работы с числовыми массивами
import numpy as np
# Импорт PIL (Python Imaging Library) для работы с изображениями
from PIL import Image
# Импорт matplotlib для отображения изображений и графиков
import matplotlib.pyplot as plt

# Функция для загрузки и подготовки изображения
def load_image(path_to_img, max_dim=512):
    # Открытие изображения по указанному пути
    img = Image.open(path_to_img)
    # Конвертация изображения в RGB (на случай, если оно в другом формате)
    img = img.convert('RGB')
    # Изменение размера изображения так, чтобы наибольшая сторона была max_dim пикселей (сохраняет пропорции)
    img.thumbnail((max_dim, max_dim))
    # Преобразование изображения в массив NumPy
    img = np.array(img)
    # Нормализация значений пикселей (0-255 -> 0-1) и добавление размерности батча (1, height, width, 3)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.
    # Преобразование массива в тензор TensorFlow
    return tf.constant(img)

# Загрузка и подготовка контентного изображения (которое будет стилизоваться)
content_image = load_image("C:/Users/PC/Desktop/gpt/content.jpg")
# Загрузка и подготовка стилевого изображения (стиль которого будет перенесен)
style_image = load_image("C:/Users/PC/Desktop/gpt/style.jpg")
# Загрузка модели стилизации изображений из TensorFlow Hub
# Эта конкретная модель — "Magenta's Arbitrary Image Stylization"
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
# Применение модели для стилизации изображения
# - content_image — изображение, которое будет стилизовано
# - style_image — изображение, стиль которого будет применен
# Результат — тензор с стилизованным изображением (берем первый элемент вывода [0])
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
# Отображение результата:
# - np.squeeze удаляет одномерные оси (в данном случае батч-размерность)
plt.imshow(np.squeeze(stylized_image))
# Отключение осей на графике
plt.axis('off')
# Установка заголовка для изображения
plt.title('Stylized Image')
# Показ изображения в окне
plt.show()
