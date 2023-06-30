from models.load_dataset import dataset_numeric_columns,dataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset_numeric_columns,
                                                    dataset["color"],
                                                    test_size=0.25,
                                                    random_state=42)
model = SVC(kernel='rbf')

model.fit(X_train, y_train)

