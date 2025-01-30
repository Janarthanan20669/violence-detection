import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

# Load the model
model_path = r'C:\Users\janar\Downloads\Violence detection system\modelnew.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_frame(frame):
    """
    Preprocess a single frame for model input.
    Assumes the model expects images of size (128, 128, 3).
    """
    # Resize frame
    frame_resized = cv2.resize(frame, (128, 128))
    # Normalize pixel values
    frame_normalized = frame_resized / 255.0
    # Expand dimensions to match model input
    return np.expand_dims(frame_normalized, axis=0)

def detect_fight(video_path):
    """
    Detects fights in a given video using the pre-trained model.

    Args:
        video_path (str): Path to the video file.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Make predictions
        prediction = model.predict(preprocessed_frame)

        # Interpret the prediction (assuming binary classification)
        if prediction[0] > 0.5:
            label = "Fight"
        else:
            label = "No Fight"

        # Display the results
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Fight Detection', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def select_video():
    """Open file dialog to select a video and run fight detection."""
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    if video_path:
        detect_fight(video_path)

# Create a modern GUI
root = tk.Tk()
root.title("Fight Detection System")
root.geometry("500x400")
root.resizable(False, False)

# Apply colorful styling
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TLabel", font=("Arial", 14))

# Add a background color
root.configure(bg="#f2f2f2")

# Create a frame for content with a border
frame = tk.Frame(root, bg="#e6f7ff", bd=5, relief="ridge")
frame.place(relx=0.5, rely=0.5, anchor="center", width=450, height=350)

# Add title label
label = tk.Label(frame, text="Fight Detection System", bg="#e6f7ff", fg="#004d99", font=("Arial", 18, "bold"))
label.pack(pady=10)

# Add an instruction label
instruction_label = tk.Label(frame, text="Select a video file to detect fights", bg="#e6f7ff", fg="#0066cc", font=("Arial", 12))
instruction_label.pack(pady=5)

# Add buttons with colors
select_button = tk.Button(frame, text="Select Video", bg="#4da6ff", fg="white", font=("Arial", 12, "bold"), command=select_video)
select_button.pack(pady=15)

exit_button = tk.Button(frame, text="Exit", bg="#ff4d4d", fg="white", font=("Arial", 12, "bold"), command=root.quit)
exit_button.pack(pady=10)

# Add a footer label
footer_label = tk.Label(frame, text="Powered by AI", bg="#e6f7ff", fg="#004d99", font=("Arial", 10, "italic"))
footer_label.pack(side="bottom", pady=10)

root.mainloop()
