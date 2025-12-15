import tkinter as tk
import pandas as pd
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x = pd.read_csv('iris_x.csv')
y = pd.read_csv('iris_y.csv')
y = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model_LogReg = LogisticRegression()
model_LogReg.fit(X_train, y_train)
LogReg_accuracy = model_LogReg.score(X_test, y_test)
print("Logistic Regression Model Accuracy:", LogReg_accuracy)
def predict_LogReg(lst):
    lst_df = pd.DataFrame([lst], columns=x.columns)
    p=model_LogReg.predict(lst_df)
    if p==0:
        return "Setosa"
    elif p==1:
        return "Versicolor"
    elif p==2:
        return "Virginica"
def predict_iris_species_LogReg():
    try:
        sepal_length = float(entry1.get())
        sepal_width = float(entry2.get())
        petal_length = float(entry3.get())
        petal_width = float(entry4.get())
        lst = [sepal_length, sepal_width, petal_length, petal_width]
        result = predict_LogReg(lst)
        result_label.config(text=f"Predicted Species: {result}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers")

window = tk.Tk()
window.title("Iris")
window.geometry("500x500")
window.configure(bg="#FFFFFF")
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
add_button = tk.Button(window, text="Species",font=("Arial", 10),fg="#FFFFFF",bg="#232F71", command=predict_iris_species_LogReg)
add_button.pack(pady=10)
result_label = tk.Label(window, text="Result: ",font=("Arial", 10),fg="#232F71", bg="#FFFFFF")
result_label.pack(pady=5)

window.mainloop()