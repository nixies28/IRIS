import tkinter as tk
from tkinter import messagebox
import Correct_SVC as svc_model
import Correct_KNN as knn_model
import Correct_LogReg as logreg_model


window = tk.Tk()
window.title("Iris")
window.geometry("1000x1000")
window.configure(bg="#FFFFFF")

model_choice = tk.StringVar()
model_choice.set("KNN")

def predict_iris_species():
    try:
        sepal_length = float(entry1.get())
        sepal_width = float(entry2.get())
        petal_length = float(entry3.get())
        petal_width = float(entry4.get())
        lst = [sepal_length,sepal_width, petal_length,petal_width]

        if model_choice.get() == "KNN":
            result = knn_model.predict_KNN(lst)
            acc = knn_model.KNN_accuracy

        elif model_choice.get() == "LogReg":
            result = logreg_model.predict_LogReg(lst)
            acc = logreg_model.LogReg_accuracy

        else:
            result = svc_model.predict_svc(lst)
            acc = svc_model.svc_accuracy

        result_label.config(text=f"Model: {model_choice.get()}\n Predicted Species: {result}\n Model Accuracy: {acc*100:.2f}%")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers")

label_header = tk.Label(window, text="Iris Species Predictor", font=("Arial", 16), fg="#232F71", bg="#FFFFFF")
label_header.pack(pady=10)
label1 = tk.Label(window, text="Sepal Length:",font=("Arial", 10),fg="#232F71", bg="#FFFFFF")
label1.pack(pady=5)
entry1 = tk.Entry(window)
entry1.pack(pady=5)

label2 = tk.Label(window, text="Sepal Width:",font=("Arial", 10),fg="#232F71", bg="#FFFFFF")
label2.pack(pady=5)
entry2 = tk.Entry(window)
entry2.pack(pady=5)

label3 = tk.Label(window, text="Petal Length:",font=("Arial", 10),fg="#232F71", bg="#FFFFFF")
label3.pack(pady=5)
entry3 = tk.Entry(window)
entry3.pack(pady=5)

label4 = tk.Label(window, text="Petal Width:",font=("Arial", 10),fg="#232F71", bg="#FFFFFF")
label4.pack(pady=5)
entry4 = tk.Entry(window)
entry4.pack(pady=5)

tk.Label(window, text="Select Model",font=("Arial", 12), bg="white", fg="#232F71").pack(pady=10)
tk.Radiobutton(window, text="KNN",variable=model_choice, value="KNN",bg="white").pack()
tk.Radiobutton(window, text="Logistic Regression",variable=model_choice, value="LogReg",bg="white").pack()
tk.Radiobutton(window, text="Support Vector Classifier",variable=model_choice, value="SVC",bg="white").pack()

add_button = tk.Button(window, text="Species",font=("Arial", 10),fg="#FFFFFF",bg="#232F71", command=predict_iris_species)
add_button.pack(pady=10)

result_label = tk.Label(window, text="Result: ",font=("Arial", 10),fg="#232F71", bg="#FFFFFF")
result_label.pack(pady=5)

window.mainloop()