from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

clf = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

grid = {"n_estimators" : [50,100,150,200,250],
       "max_depth": [None, 10, 20],
       "min_samples_split": [2, 5],
       "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=grid,
    cv=4,  # correct usage
    n_jobs=-1,
    scoring="accuracy",
    verbose=2
)

grid_search.fit(X_train,y_train)
print("Best Parameters:", grid_search.best_params_)


best_model = grid_search.best_estimator_

#predict
y_pred_b = best_model.predict(X_test)

print('Tuned Accuracy:', accuracy_score(y_pred,y_test))
print("\nClassification Report:", classification_report(y_test,y_pred,target_names=le.classes_))
