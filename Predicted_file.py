import pickle
from sklearn.metrics import mean_squared_error
from creating_model import X_test, Y_test
load_model = pickle.load(open('Lasso_model_sav','rb'))
y_predict = load_model.predict(X_test)
error_val = mean_squared_error(y_true=Y_test,y_pred=y_predict)
print('MSE of Lasso: ',error_val)
# accuracy = accuracy_score(y_true=Y_test,y_pred=y_predict)
# print('Accuracy of Lasso: ',accuracy*100)
print('#############################')
load_model = pickle.load(open('Elasticnet_model_sav','rb'))
y_predict = load_model.predict(X_test)
error_val = mean_squared_error(y_true=Y_test,y_pred=y_predict)
print('MSE of Elastic Net: ',error_val)
# accuracy = accuracy_score(y_true=Y_test,y_pred=y_predict)
# print('Accuracy of Elastic Net: ',accuracy*100)


