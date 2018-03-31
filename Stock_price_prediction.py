import csv
import numpy
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# def find_sqft_range_map(price_sqft):
#     if price_sqft >= 500 and price_sqft < 1000:
#         return 1
#     elif price_sqft >= 1000 and price_sqft < 1500:
#         return 2
#     elif price_sqft >= 1500 and price_sqft < 2000:
#         return 3
#     elif price_sqft >= 2000 and price_sqft <2500:
#         return 4
#     elif price_sqft >=2500 and price_sqft <3000:
#         return 5
#     elif price_sqft >= 3000 and price_sqft <4000:
#         return 6
#     else:
#         return 7

# def find_price_range_map(price_sqft):
#     if price_sqft >= 500000 and price_sqft < 1000000:
#         return 1
#     elif price_sqft >= 1000000 and price_sqft < 1500000 :
#         return 2
#     elif price_sqft >= 1500000 and price_sqft < 2000000:
#         return 3
#     elif price_sqft >= 2000000 and price_sqft <2500000:
#         return 4
#     elif price_sqft >=2500000 and price_sqft <3000000:
#         return 5
#     elif price_sqft >= 3000000 and price_sqft <4000000:
#         return 6
#     else:
#         return 7



read_file_name = 'stockbangladesh.com_01-04-2013.csv'
train_data=[]
train_target=[]
with open(read_file_name, 'rt') as csv_file:
    count = 0
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
    for row in csv_reader:
        if count > 0  :
              temp=[]
              temp.append(float(row[2]))
              temp.append(float(row[3]))
              temp.append(float(row[4]))
              temp.append(float(row[6]))
              train_data.append(temp)
              train_target.append(float(row[5]))

        count+=1


        # for testing data
test_data=[]
test_target=[]
test_file_name='stockbangladesh.com_01-01-2013.csv'
with open(test_file_name, 'rt') as csv_file:
    count = 0
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
    for row in csv_reader:
        if count > 0:
              predict=[]
              predict.append(float(row[2]))
              predict.append(float(row[3]))
              predict.append(float(row[4]))
              predict.append(float(row[6]))
              test_data.append(predict)
              test_target.append(float(row[5]))

        count+=1

'''
####### nearest neighbor algorithm
neigh = KNeighborsRegressor(n_neighbors=10)
neigh.fit(train_data, train_target)
predicted_price = neigh.predict(test_data)
# print(predicted_price)
error = 0.0
for i in range(0, len(predicted_price)):
    error += (abs(test_target[i] - predicted_price[i]) / test_target[i])
error = error / len(predicted_price)
accuracy = 100.0 - (error * 100.0)
print("Predicted Apartment price :"+str(accuracy))

'''
'''
####### decision tree
decision_tree = DecisionTreeRegressor(max_depth=5)
decision_tree.fit(train_data, train_target)
predicted_price = decision_tree.predict(test_data)
error = 0.0
for i in range(0, len(predicted_price)):
    error += (abs(test_target[i] - predicted_price[i]) / test_target[i])
error = error / len(predicted_price)
# print(error)
accuracy = 100.0 - (error * 100.0)
print("Predicted Apartment price :"+str(accuracy))
'''

####### multi layer perceptron
multi_perceptron = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1,
                                activation='logistic')
multi_perceptron.fit(train_data, train_target)
predicted_price = multi_perceptron.predict(test_data)
error = 0.0
for i in range(0, len(predicted_price)):
    error += (abs(test_target[i] - predicted_price[i]) / test_target[i])
error = error / len(predicted_price)
accuracy = 100.0 - (error * 100.0)
print("Predicted Apartment price :"+str(accuracy))

