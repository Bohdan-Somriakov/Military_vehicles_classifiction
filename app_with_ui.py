import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import PhotoImage
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tkinter import Image as TkImage
from PIL import Image as PILImage, ImageTk


class MilitaryVehicleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Military Vehicle Classification")
        self.root.geometry("600x500")

        self.train_dir = None
        self.test_dir = None
        self.model = None
        self.history = None
        self.uploaded_image_path = None

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Military Vehicle Classification App", font=("Arial", 16)).pack(pady=10)

        self.train_button = ttk.Button(self.root, text="Select Training Dataset", command=self.select_train_dir)
        self.train_button.pack(pady=5)

        self.test_button = ttk.Button(self.root, text="Select Testing Dataset", command=self.select_test_dir)
        self.test_button.pack(pady=5)

        self.train_model_button = ttk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)

        self.test_model_button = ttk.Button(self.root, text="Test Model", command=self.test_model, state=tk.DISABLED)
        self.test_model_button.pack(pady=10)

        self.load_model_button = ttk.Button(self.root, text="Load Pre-trained Model (H5)",
                                            command=self.load_model_from_file)
        self.load_model_button.pack(pady=5)

        self.upload_image_button = ttk.Button(self.root, text="Upload Image for Prediction", command=self.upload_image)
        self.upload_image_button.pack(pady=5)

        self.predict_button = ttk.Button(self.root, text="Predict", command=self.predict_image, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.status_label = ttk.Label(self.root, text="Status: Waiting for user input", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.prediction_label = ttk.Label(self.root, text="Prediction: None", font=("Arial", 12))
        self.prediction_label.pack(pady=10)

        self.image_label = ttk.Label(self.root, text="No image uploaded", font=("Arial", 12))
        self.image_label.pack(pady=10)

    def select_train_dir(self):
        self.train_dir = filedialog.askdirectory(title="Select Training Dataset Directory")
        if self.train_dir:
            messagebox.showinfo("Directory Selected", f"Training dataset selected:\n{self.train_dir}")
            self.status_label.config(text="Status: Training dataset loaded")

    def select_test_dir(self):
        self.test_dir = filedialog.askdirectory(title="Select Testing Dataset Directory")
        if self.test_dir:
            messagebox.showinfo("Directory Selected", f"Testing dataset selected:\n{self.test_dir}")
            self.status_label.config(text="Status: Testing dataset loaded")

    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same', input_shape=(256, 256, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(64, (6, 6), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self):
        if not self.train_dir:
            messagebox.showerror("Error", "Please select a training dataset directory first.")
            return

        train_data_gen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_data_gen.flow_from_directory(
            directory=self.train_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical'
        )

        self.create_model()

        self.status_label.config(text="Status: Training the model...")

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=10
        )

        self.status_label.config(text="Status: Model training completed")
        messagebox.showinfo("Training Completed", "Model training completed.")
        self.test_model_button.config(state=tk.NORMAL)

    def test_model(self):
        if not self.test_dir:
            messagebox.showerror("Error", "Please select a testing dataset directory first.")
            return

        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            directory=self.test_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical'
        )

        self.status_label.config(text="Status: Testing the model...")
        results = self.model.evaluate(test_generator)
        accuracy = results[1]
        self.status_label.config(text=f"Status: Testing completed. Accuracy: {accuracy:.2%}")
        messagebox.showinfo("Testing Completed", f"Model accuracy: {accuracy:.2%}")

    def load_model_from_file(self):
        model_file = filedialog.askopenfilename(filetypes=[("H5 Files", "*.h5")])
        if model_file:
            self.model = load_model(model_file)
            self.status_label.config(text="Status: Pre-trained model loaded successfully.")
            messagebox.showinfo("Model Loaded", "Pre-trained model loaded successfully.")
            self.test_model_button.config(state=tk.NORMAL)

    def upload_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            img = PILImage.open(image_path).resize((256, 256))

            img_display = ImageTk.PhotoImage(img)

            self.image_label.config(image=img_display)
            self.image_label.image = img_display 

            self.uploaded_image_path = image_path

            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)  

            class_names = sorted(os.listdir(self.train_dir))  
            predicted_label = class_names[predicted_class_index[0]]  

            self.status_label.config(text=f"Status: Image classified as {predicted_label}")
            self.prediction_label.config(text=f"Prediction: {predicted_label}")

            self.predict_button.config(state=tk.NORMAL)

    def predict_image(self):
        if not self.uploaded_image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        img = PILImage.open(self.uploaded_image_path).resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        class_names = os.listdir(self.train_dir)
        predicted_label = class_names[predicted_class[0]]

        self.status_label.config(text=f"Status: Image classified as {predicted_label}")
        self.prediction_label.config(text=f"Prediction: {predicted_label}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MilitaryVehicleApp(root)
    root.mainloop()
