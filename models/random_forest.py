from sklearn.ensemble import RandomForestClassifier
from models.load_dataset import X_train, y_train
model = RandomForestClassifier()

model.fit(X_train, y_train)