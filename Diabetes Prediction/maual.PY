import numpy as np
import warnings
import pickle
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Define feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction', 'Age']

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise Exception("Model file not found. Please check the path and try again.")


# Function to make predictions
def predict_diabetes():
    input_data = []
    for entry in entries:
        try:
            value = float(entry.get())
            if value < 0:
                raise ValueError
            input_data.append(value)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid non-negative numeric values for all fields.")
            return

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    try:
        prediction = model.predict(input_data_as_numpy_array)
    except Exception as e:
        messagebox.showerror("Prediction error", f"An error occurred during prediction: {e}")
        return

    result = 'The person does not have diabetes.' if prediction[0] == 0 else 'The person has diabetes.'
    messagebox.showinfo("Diabetes Prediction", result)


# Function to clear all inputs
def clear_inputs():
    for entry in entries:
        entry.delete(0, tk.END)


# Set up the GUI
root = tk.Tk()
root.title("Diabetes Prediction System by Smith Chauhan@interpe")
root.geometry("1000x400")  # Adjust the initial window size
root.resizable(True, True)  # Make the window resizable

# Create a main frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a canvas
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

# Create another frame inside the canvas
frame = tk.Frame(canvas, bg="#2e2e2e")

# Add the new frame to a window in the canvas
canvas.create_window((0, 0), window=frame, anchor="nw")

header = tk.Label(frame, text="Diabetes Prediction System", font=("Helvetica", 24, "bold"), bg="#2e2e2e", fg="#ffffff")
header.grid(row=0, column=0, columnspan=5, pady=20)

entries = []
for idx, feature_name in enumerate(feature_names):
    row = 1 if idx < 5 else 3
    column = idx if idx < 5 else idx - 5

    label = tk.Label(frame, text=f"{feature_name}:", font=("Helvetica", 14), bg="#2e2e2e", fg="#ffffff", anchor='w')
    label.grid(row=row, column=column, padx=5, pady=5, sticky="w")

    entry = tk.Entry(frame, font=("Helvetica", 14), bg="#ffffff", fg="#000000")
    entry.grid(row=row + 1, column=column, padx=5, pady=5, sticky="w")
    entries.append(entry)

button_frame = tk.Frame(frame, bg="#2e2e2e")
button_frame.grid(row=5, column=0, columnspan=5, pady=20)

submit_button = tk.Button(button_frame, text="Predict", command=predict_diabetes, font=("Helvetica", 14), bg="#4CAF50",
                          fg="white", padx=20, pady=10, bd=0, relief=tk.RIDGE, highlightthickness=0)
submit_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_inputs, font=("Helvetica", 14), bg="#f44336",
                         fg="white", padx=20, pady=10, bd=0, relief=tk.RIDGE, highlightthickness=0)
clear_button.pack(side=tk.LEFT, padx=10)

footer = tk.Label(frame, text="Powered by Smith Chauhan @InternPe", font=("Helvetica", 12), bg="#2e2e2e", fg="#888888")
footer.grid(row=6, column=0, columnspan=5, pady=10)

root.mainloop()
