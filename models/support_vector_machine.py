from load_dataset import X_train, y_train
from sklearn.svm import SVC

model = SVC(kernel='rbf')

model.fit(X_train, y_train)



