import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_squared_error

# Функція обчислення зміщення між двома кадрами
def calculate_displacement(image1, image2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    displacement = np.mean([np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt) for m in matches], axis=0)
    return displacement

# Завантаження кадрів
frames = [cv2.imread(f'frame{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 5)]

# Розрахунок зсувів між кадрами
displacements = [calculate_displacement(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]

print("Зсуви між кадрами:", displacements)

# Підготовка даних для навчання
x_train = np.array([[0, 0]] + displacements[:-1])
x_train = np.expand_dims(x_train, axis=1)  # Додаємо вимір для LSTM

y_train = np.array(displacements)

# Побудова LSTM моделі
model = Sequential([
    Input(shape=(1, 2)),
    LSTM(50, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Навчання моделі
model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=2)

# Прогноз наступного зсуву
predicted_displacement = model.predict(np.array([displacements[-1]]).reshape(1, 1, 2))
print(f"Прогнозований зсув: {predicted_displacement}")

# Розрахунок похибки
actual_displacement = displacements[-1] + displacements[-2]
mse = mean_squared_error(actual_displacement, predicted_displacement.flatten())
print(f"Середньоквадратична похибка: {mse}")

# Функція генерації CSS-анімації
def generate_css_animation(displacement_sequence, name, timing_function):
    keyframes = ""
    for index, displacement in enumerate(displacement_sequence):
        percentage = (index * 100) / (len(displacement_sequence) - 1)
        keyframes += f"""
            {percentage}% {{
                transform: translate({displacement[0]}px, {displacement[1]}px);
            }}
        """
    return f"""
        @keyframes {name} {{
            {keyframes}
        }}

        .{name}-object {{
            animation: {name} 4s {timing_function} infinite;
        }}
    """

# Підготовка зсувів для CSS
displacement_sequence = [[0, 0]] + displacements
predicted_sequence = displacement_sequence + [predicted_displacement.flatten()]

# Генерація CSS з різними функціями часу
original_css = generate_css_animation(displacement_sequence, "original", "linear")
predicted_css = generate_css_animation(predicted_sequence, "predicted", "cubic-bezier(0.42, 0, 0.58, 1)")

# Створення HTML для порівняння анімацій
html_content = f"""
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Animation Comparison</title>
    <style>
        {original_css}
        {predicted_css}

        body {{
            text-align: center;
            font-family: Arial, sans-serif;
        }}

        .container {{
            display: flex;
            justify-content: center;
            gap: 50px;
        }}

        .box {{
            width: 300px;
            height: 300px;
            background-color: lightgray;
            position: relative;
            overflow: hidden;
        }}

        #first-block, #second-block {{
            border: 2px solid black;
            padding: 20px;
        }}

        .square {{
            width: 50px;
            height: 50px;
            background-color: blue;
            position: absolute;
            animation: none;
        }}

        .controls{{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Аналіз анімацій за допомогою LSTM</h1>
    <div class="container">
        <div id="first-block">
            <h3>Оригінальна анімація</h3>
            <div class="box" id="original-box">
                <div class="square" id="original-square"></div>
            </div>
        </div>
        <div id="second-block">
            <h3>Передбачена анімація</h3>
            <div class="box" id="predicted-box">
                <div class="square" id="predicted-square"></div>
            </div>
        </div>
    </div>
    <div class="controls">
        <button onclick="startAnimation()">Відтворити</button>
        <button onclick="pauseAnimation()">Пауза</button>
        <button onclick="resetAnimation()">Скинути</button>
        <button onclick="changeAnimationType()">Змінити тип анімації</button>
    </div>

    <script>
        const animationTypes = [
            "linear",
            "ease",
            "ease-in",
            "ease-out",
            "ease-in-out",
            "cubic-bezier(0.42, 0, 0.58, 1)",
            "cubic-bezier(0.25, 1, 0.5, 1)"
        ];
        let currentAnimationIndex = 0;

        function startAnimation() {{
            document.querySelector("#original-square").style.animation = "original 4s " + animationTypes[currentAnimationIndex] + " infinite";
            document.querySelector("#predicted-square").style.animation = "predicted 4s " + animationTypes[currentAnimationIndex] + " infinite";
        }}

        function pauseAnimation() {{
            document.querySelector("#original-square").style.animationPlayState = 'paused';
            document.querySelector("#predicted-square").style.animationPlayState = 'paused';
        }}

        function resetAnimation() {{
            document.querySelector("#original-square").style.animation = 'none';
            document.querySelector("#predicted-square").style.animation = 'none';
            setTimeout(startAnimation, 50);
        }}

        function changeAnimationType() {{
            currentAnimationIndex = (currentAnimationIndex + 1) % animationTypes.length;
            console.log("Новий тип анімації: " + animationTypes[currentAnimationIndex]);
            resetAnimation();
        }}
    </script>
</body>
</html>
"""

# Збереження HTML-файлу
with open("animation_comparison.html", "w", encoding='utf-8') as file:
    file.write(html_content)

print("HTML з двома анімаціями збережено як 'animation_comparison.html'")
