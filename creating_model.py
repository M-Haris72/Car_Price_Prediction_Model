import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas.core.dtypes.common import classes
from scipy.stats import alpha
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, r2_score
import pickle


# Using pandas Library to read a CSV file
data = pd.read_csv('car_price_dataset.csv')
# info() give us a information about datafram(Rows & Columns)
print(data.info())
# describe give info about us a mean ,max ,min, 25% ,50% ,75%
print(data.describe())
# isnull() method is check some data is missing? and sum() use to add up which tell us how many enteries are null
print(data.isnull().sum())
#  Columns give us  all column names of data
print(data.columns)


# unique() method is use to check the unique values in a column
print(data['Model'].unique())

# regplot() in seaborn library regression plot is used to find correlation between two variables is that a dependent or not # para data=your_data x='column_name on x axis' y= 'cloumn_name on y axis' compare only numerical columns
# sb.regplot(data= data,x='Mileage', y='Price' , color='blue')


# # seperate the non_numerical data from the numerical data to check heatmap() relationship between variables
non_numerical = data.select_dtypes(include=['object']).columns
# print(non_numerical)

# drop() method will use to drop the columns from the dataset which are passed in non_numerical and also para axis = 1
numerical_data = data.drop(non_numerical, axis=1)
# print(numerical_data.head())
#
# heatmap() in seaborn which used to visual representation of columns relationship
sb.heatmap(data= numerical_data.corr(), annot=True )
plt.show()

# Scaling the numerical data for range(0-1) use the scikit learn library minmaxscaler
# initialize it minmaxscaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numerical_data)
scaled_data = pd.DataFrame(scaled_data)


# Encoded the non_numerical feature of data
unencoded_data = {}
for column_name in non_numerical:
    unencoded_data[column_name] = data.pop(column_name)
unencoded_data = pd.DataFrame(unencoded_data)
# print(unencoded_data.head())



# using scikit learn library encoded the data which is given in category or other types
label_encoder = LabelEncoder()
# label_data dict use to store the encoded data after that we will convert it into dataframe
encoded_data = {}
# it will use to map in future when make application for prediction so we can see the label has value
label_mapping = {}
for column_name in unencoded_data:
    encoded_data[column_name] = label_encoder.fit_transform(unencoded_data[column_name])
    # we will use  dict to store the multi columns encoding becoz otherwise LabelEncoder() only provide last encoding column
    # zip() function combine both list of label and encoding number
    label_mapping[column_name] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
encoded_data = pd.DataFrame(encoded_data)
# now encoded data convert it into min max scaling using scikit learn
scaled_encoded_data = scaler.fit_transform(encoded_data)
scaled_encoded_data = pd.DataFrame(scaled_encoded_data)
# print(scaled_data)
# print(scaled_encoded_data)

# combine tha data frame using concat in pandas lab of scaled_data in which has only numerical values while scaled_encoded_data has non numermical values and check the variables has more impact on predicted value
combine_df = pd.concat([scaled_data,scaled_encoded_data], axis= 1)
sb.heatmap(combine_df.corr(),annot=True)
plt.show()


x_dataset = scaled_data.drop(columns=[5])
y_dataset = numerical_data.pop('Price')
# split the data into testing and training data with using sklearn library
X_train , X_test, Y_train, Y_test = train_test_split(x_dataset,y_dataset,test_size=0.3,train_size=0.7,random_state=50 ,shuffle=True)
lasso = Lasso(alpha=0.001)
lasso_model = lasso.fit(X_train,Y_train)

elasticnet = ElasticNet(alpha=1.0,max_iter=1000,random_state=32)
elasticnet_model = elasticnet.fit(X_train,Y_train)
#  
filename = 'Lasso_model_sav'
pickle.dump(lasso_model,open(filename,'wb'))

filename = 'Elasticnet_model_sav'
pickle.dump(elasticnet_model,open(filename,'wb'))

print('END')

# y_predict = lasso_model.predict(X_test)
# print(r2_score(y_true=Y_test, y_pred=y_predict))


# y_axis = []
# for a in range(3000):
#     y_axis.append(a)
#
# plt.figure(figsize=(10,10))
# # sb.barplot(x = y_predict, y = Y_test, color = 'Red')
#
# plt.plot(y_axis,Y_test,y_axis,y_predict)
# plt.title('Variance between Predict and Actual Value', fontsize = 0.5)
# plt.xlabel('y test and y predict')
# plt.show()
