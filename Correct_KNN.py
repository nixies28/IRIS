from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

x = pd.read_csv('iris_x.csv')
y = pd.read_csv('iris_y.csv')
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model_KNN = KNeighborsClassifier(n_neighbors=3)
model_KNN.fit(X_train, y_train)

KNN_accuracy = model_KNN.score(X_test, y_test)

def predict_KNN(lst):
    lst_df = pd.DataFrame([lst], columns=x.columns)
    p=model_KNN.predict(lst_df)
    if p==0:
        return "Setosa"
    elif p==1:
        return "Versicolor"
    elif p==2:
        return "Virginica"

# def predict_iris_species_KNN():
#     try:
#         sepal_length = float(entry1.get())
#         sepal_width = float(entry2.get())
#         petal_length = float(entry3.get())
#         petal_width = float(entry4.get())

#         lst = [sepal_length,sepal_width, petal_length,petal_width]

#         result = predict_KNN(lst)
#         result_label.config(text=f"Predicted Species: {result}")

#     except ValueError:
#         messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields")