import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
from playsound import playsound
import threading
import customtkinter as ctk
import os
from datetime import datetime
from collections import deque
import sqlite3

# ---------------- Overlay Image ----------------
def overlay_image(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if x + ow > bw or y + oh > bh:
        overlay = overlay[0:max(0, bh - y), 0:max(0, bw - x)]

    for c in range(0, 3):
        background[y:y+oh, x:x+ow, c] = (
            overlay[:, :, c] * (overlay[:, :, 3] / 255.0)
            + background[y:y+oh, x:x+ow, c] * (1.0 - overlay[:, :, 3] / 255.0)
        )

# -------- Config --------
CAPTURE_DURATION = 10
SAVE_DEBUG_IMAGES = False
alert_display_start_time = None
show_alert_message = False

# -------- SQLite DB Setup --------
DB_FILE = "drowsiness_alerts"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_alert(status):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO alerts (timestamp, status) VALUES (?, ?)", (timestamp, status))
    conn.commit()
    conn.close()

def get_all_alerts():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

# -------- Model Definition --------
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# -------- Load Model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessCNN().to(device)
model.load_state_dict(torch.load("./model/drowsiness_cnn.path", map_location=device))
model.eval()

# -------- Image Transform --------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# -------- Mediapipe Setup --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# -------- Alert Function --------
def alert_user():
    global show_alert_message
    show_alert_message = True
    threading.Thread(target=playsound, args=("alert.mp3",), daemon=True).start()

# -------- Save Debug Images --------
def save_debug_image(folder_name, eye_img):
    if not SAVE_DEBUG_IMAGES:
        return
    os.makedirs(folder_name, exist_ok=True)
    timestamp = datetime.now().strftime("Y%m%d_%H%M%S_%f")
    filename = os.path.join(folder_name, f"eye_{timestamp}.png")
    cv2.imwrite(filename, eye_img)

# -------- Start Detection --------
def start_detection():
    global show_alert_message, alert_display_start_time

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    buffer_size = int(fps * CAPTURE_DURATION)
    print(f"Buffer size for {CAPTURE_DURATION}s: {buffer_size} frames")

    left_eye_pts = [33, 133, 160, 158, 157, 173, 246]
    right_eye_pts = [362, 263, 387, 386, 385, 384, 398, 466]
    prediction_buffer = deque(maxlen=buffer_size)

    mp_face_mesh_proc = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    siren_img = cv2.imread("siren.png", cv2.IMREAD_UNCHANGED)
    if siren_img is None:
        raise FileNotFoundError("Could not load siren image. Make sure 'siren.png' exists.")
    if siren_img.shape[2] == 3:
        b, g, r = cv2.split(siren_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        siren_img = cv2.merge([b, g, r, alpha])
    elif siren_img.shape[2] != 4:
        raise ValueError("Siren image must have 4 channels (BGRA) for transparency.")

    siren_img = cv2.resize(siren_img, (50, 50))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh_proc.process(rgb)

        status = "Face Not Detected"
        preds = []

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark

            def extract_eye(eye_indices):
                x_coords = [int(landmarks[i].x * w) for i in eye_indices]
                y_coords = [int(landmarks[i].y * h) for i in eye_indices]
                x_min, x_max = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, w)
                y_min, y_max = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, h)
                eye_img = frame[y_min:y_max, x_min:x_max]
                if eye_img.shape[0] < 10 or eye_img.shape[1] < 10:
                    return None
                return cv2.resize(eye_img, (64, 64))

            left_eye_img = extract_eye(left_eye_pts)
            right_eye_img = extract_eye(right_eye_pts)

            for eye_img, label in zip([left_eye_img, right_eye_img], ['left', 'right']):
                if eye_img is not None:
                    save_debug_image(f"./debug_eyes/{label}", eye_img)
                    eye_pil = Image.fromarray(eye_img)
                    input_tensor = transform(eye_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        pred = torch.argmax(output, 1).item()
                        preds.append(pred)

            if preds:
                avg_pred = round(sum(preds) / len(preds))
                prediction_buffer.append(avg_pred)
                status = "Eyes Closed" if avg_pred == 0 else "Eyes Open"
            else:
                status = "Eyes Undetected"

        # Drowsiness Check
        if len(prediction_buffer) == buffer_size:
            closed_ratio = prediction_buffer.count(0) / buffer_size
            print(f"Closed ratio: {closed_ratio:.2f}")
            if closed_ratio >= 0.7:
                status = "DROWSY - ALERT!"
                alert_user()
                alert_display_start_time = datetime.now()
                prediction_buffer.clear()

                # Log alert to SQLite DB
                log_alert(status)

        color = (0, 255, 0) if "Open" in status else (0, 0, 255)
        cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if show_alert_message:
            cv2.putText(frame, "You seem sleepy, Please don't drive!",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            overlay_image(frame, siren_img, x=frame.shape[1] - 60, y=10)
            if alert_display_start_time and (datetime.now() - alert_display_start_time).total_seconds() > 3:
                show_alert_message = False
                alert_display_start_time = None

        cv2.imshow("Driver Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting detection...")
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- GUI --------
def show_alerts_gui():
    rows = get_all_alerts()

    window = ctk.CTkToplevel()
    window.title("Saved Alerts")
    window.geometry("500x300")

    text_box = ctk.CTkTextbox(window, wrap="none", font=("Arial", 12))
    text_box.pack(fill="both", expand=True, padx=10, pady=10)

    if not rows:
        text_box.insert("end", "No alerts found.")
    else:
        for row in rows:
            text_box.insert("end", f"ID: {row[0]} | Time: {row[1]} | Status: {row[2]}\n")

def run_gui():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")

    app = ctk.CTk()
    app.geometry("500x400")
    app.title("Driver Drowsiness Detection")

    title = ctk.CTkLabel(app, text="Drowsiness Detection", font=("Arial", 16))
    title.pack(pady=20)

    start_button = ctk.CTkButton(app, text="Start Detection", command=start_detection, font=("Arial", 16))
    start_button.pack(pady=10)

    alerts_button = ctk.CTkButton(app, text="View Saved Alerts", command=show_alerts_gui, font=("Arial", 16))
    alerts_button.pack(pady=10)

    quit_button = ctk.CTkButton(app, text="Exit", command=app.destroy, font=("Arial", 16))
    quit_button.pack(pady=10)

    app.mainloop()

if __name__ == "__main__":
    init_db()
    run_gui()
