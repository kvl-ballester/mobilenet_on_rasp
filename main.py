import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# --- Configuración ---
TFLITE_MODEL_PATH = "mobilenet_v2.tflite"
LABELS_PATH = "ImageNetLabels.txt"
TARGET_SIZE = 224
SCALE_SIZE = 256
FPS = 5

# --- Preprocesado como MobileNetV2 (modo "tf") ---
def preprocess_image(frame):
    h, w, _ = frame.shape
    scale = SCALE_SIZE / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Center crop a 224x224
    start_x = (new_w - TARGET_SIZE) // 2
    start_y = (new_h - TARGET_SIZE) // 2
    cropped = resized[start_y:start_y + TARGET_SIZE, start_x:start_x + 
TARGET_SIZE]

    # BGR -> RGB y escala a [-1, 1]
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32)
    img = (img / 127.5) - 1.0

    return np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

# --- Cargar etiquetas ---
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0].lower() == "background":
    labels = labels[1:]  # MobileNetV2 suele tener 1000 clases sin 
background

# --- Cargar modelo TFLite ---
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Captura de webcam e inferencia ---
cap = cv2.VideoCapture(0)
delay = 1.0 / FPS

print("Presiona Ctrl+C para salir\n")

try:
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo de la cámara")
            break

        input_tensor = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        top_indices = output.argsort()[-3:][::-1]  # Top 3
        print("\nTop 3 predicciones:")
        for i in top_indices:
            print(f"{labels[i]:<25} - {output[i]:.4f}")

        elapsed = time.time() - start
        time.sleep(max(0, delay - elapsed))

except KeyboardInterrupt:
    print("\nInterrumpido por el usuario")

cap.release()

