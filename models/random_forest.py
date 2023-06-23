# Crear el modelo de Random Forest
randomforest_model = RandomForestClassifier()

# Ajustar el modelo con los datos de entrenamiento
randomforest_model.fit(X_train, y_train)

# Visualizar la importancia de las variables
importance = randomforest_model.feature_importances_
print(importance)
