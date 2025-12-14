from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

x = pd.read_csv('iris_x.csv')
y = pd.read_csv('iris_y.csv')
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model_svc = SVC()
model_svc.fit(X_train, y_train)

svc_accuracy = model_svc.score(X_test, y_test)

def predict_svc(lst):
    lst_df = pd.DataFrame([lst], columns=x.columns)
    p=model_svc.predict(lst_df)
    if p==0:
        return "Setosa"
    elif p==1:
        return "Versicolor"
    elif p==2:
        return "Virginica"

# def predict_iris_species_SVC():
#     try:
#         sepal_length = float(entry1.get())
#         sepal_width = float(entry2.get())
#         petal_length = float(entry3.get())
#         petal_width = float(entry4.get())

#         lst = [sepal_length,sepal_width, petal_length,petal_width]

#         result = predict_svc(lst)
#         result_label.config(text=f"Predicted Species: {result}")

#     except ValueError:
#         messagebox.showerror("Invalid Input", "Please enter valid numbers")