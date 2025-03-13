import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_squared_error

# Функція для обчислення зміщення між двома зображеннями
def calculate_displacement(image1, image2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    displacement = np.mean([np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt) for m in matches], axis=0)
    return displacement

# Завантаження зображень
image1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('frame3.png', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('frame4.png', cv2.IMREAD_GRAYSCALE)

# Обчислення зміщення між кадрами
displacement1 = calculate_displacement(image1, image2)
displacement2 = calculate_displacement(image2, image3)
displacement3 = calculate_displacement(image3, image4)

print(f"Displacement between frame1 and frame2: {displacement1}")
print(f"Displacement between frame2 and frame3: {displacement2}")
print(f"Displacement between frame3 and frame4: {displacement3}")

# Формуємо тренувальні дані
x_train = np.array([[0, 0], displacement1, displacement2])
x_train = np.expand_dims(x_train, axis=1)  # Додаємо розмірність для LSTM

y_train = np.array([displacement1, displacement2, displacement3])

# Створення моделі для прогнозування зміщення
model = Sequential()
model.add(Input(shape=(1, 2)))  # Вхід для послідовності зміщень
model.add(LSTM(50, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mean_squared_error')

# Тренування моделі
model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=2)

# Прогнозування зміщення для наступного кадру
predicted_displacement = model.predict(np.array([displacement3]).reshape(1, 1, 2))
print(f"Predicted displacement for the next frame: {predicted_displacement}")

# Оцінка точності передбачення
actual_displacement = np.array([displacement2[0] + displacement3[0], displacement2[1] + displacement3[1]])  # Очікуване зміщення
mse = mean_squared_error(actual_displacement, predicted_displacement.flatten())
print(f"Mean Squared Error between predicted and actual displacement: {mse}")

# Генерація CSS анімації з використанням кривих Безьє
def generate_css_animation_with_bezier(displacement_sequence):
    keyframes = ''
    for index, displacement in enumerate(displacement_sequence):
        percentage = (index * 100) / (len(displacement_sequence) - 1)
        keyframes += f"""
            {percentage}% {{
                transform: translate({displacement[0]}px, {displacement[1]}px);
            }}
        """
        
    # Використовуємо cubic-bezier для створення плавних кривих руху
    css_animation = f"""
        @keyframes moveObject {{
            {keyframes}
        }}

        .moving-object {{
            animation: moveObject 4s cubic-bezier(0.42, 0, 0.58, 1) infinite;
        }}
    """
    return css_animation

# Створення послідовності зміщень для анімації
displacement_sequence = [
    [0, 0],
    displacement1,
    displacement2,
    displacement3
]

css_code = generate_css_animation_with_bezier(displacement_sequence)
print("Generated CSS Animation with Bezier Curves:")
print(css_code)

# Генерація HTML файлу
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Animation Test</title>
    <style>
        {css_code}
    </style>
</head>
<body>
    <div class="moving-object" style="width: 50px; height: 50px; background-color: red; position: absolute;"></div>
</body>
</html>
"""

# Збереження HTML файлу
with open("optimized_animation_test.html", "w") as file:
    file.write(html_content)

print("Optimized HTML file with Bezier animation saved as 'optimized_animation_test.html'")
