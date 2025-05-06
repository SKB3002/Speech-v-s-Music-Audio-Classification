from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

svm_classifier = SVC(kernel='rbf',C=1.0, gamma = 'scale', random_state = 42)

svm_classifier.fit(X_train_scaled,y_train)

y_pred_svm = svm_classifier.predict(X_test_scaled)

print("SVM Accuracy Score:", accuracy_score(y_test,y_pred_svm))
print("\nSVM Classification Report:",'\n',classification_report(y_test,y_pred_svm,target_names=le.classes_))
