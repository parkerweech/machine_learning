from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from numpy import sqrt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import KFold

def readData(link):
    data_set = pd.read_csv(link,header=None)
    return data_set
    
def encodeData(data_set,encoder):
    for col in data_set:
        if is_string_dtype(data_set[col]):
            encoder.fit(data_set[col])
            data_set[col] = encoder.transform(data_set[col])
    return data_set 
    
def kNN(X_test, X_train, y_test, y_train):
    
    targets_predicted = []
    i = 0
    
    done = False
    
    while done == False:
        k = int(input("Please enter a value for k: "))
        if (k < 1 or k > len(X_train)):
            print("Value of k must be greater than zero and less than the quantity of"
                  " data values, which is", len(X_train), "\nPlease try again.")
        else:
            done = True
    
    for row in X_test:
        neighbors = []
        closest = []
        j = 0
        
        for row in X_train:
            dist = 0

            for x in range(0,X_test.shape[1]):
                num1 = int(X_test[i][x])
                num2 = int(X_train[j][x])
                dist += (num1 + num2)**2
                       
            dist = sqrt(dist)
            
            type = y_train[j]
            
            data_point = DataPoint(dist,type)
            neighbors.append(data_point)
            
            j += 1
        
        neighbors.sort(key=lambda data_point: data_point.dist)
        
        l = 0
        
        zero_count = 0
        one_count = 0
        two_count = 0
        
        for x in range(0,k):
            if neighbors[l].type == 0:
                zero_count += 1
            if neighbors[l].type == 1:
                one_count += 1
            if neighbors[l].type == 2:
                two_count += 1
            l += 1  
            
        if zero_count > one_count and zero_count > two_count:
            targets_predicted.append(0)
        elif one_count > two_count and one_count > zero_count:
            targets_predicted.append(1)
        elif two_count > zero_count and two_count > one_count:
            targets_predicted.append(2)
        elif zero_count == one_count and zero_count > two_count:
            targets_predicted.append(0)
        elif one_count == two_count and one_count > zero_count:
            targets_predicted.append(1)
        elif zero_count == two_count and zero_count > one_count:
            targets_predicted.append(2)
        
        i += 1
        
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(X_train,y_train)
    predictions = model.predict(X_test)    
                
    number_correct = 0
    number_correct_2 = 0
    total = len(targets_predicted)
    index = 0
    
    
    for num in targets_predicted:
        if targets_predicted[index] == y_test[index]:
            number_correct += 1
        if predictions[index] == y_test[index]:
            number_correct_2 += 1
        index += 1
    
    percent_accurate = number_correct / total * 100
    percent_accurate_2 = number_correct_2 / total * 100
        
    print("Our function has a {:.1f}% accuracy.".format(percent_accurate))
    print("The built in function has a {:.1f}% accuracy.".format(percent_accurate_2)) 
         
    
        
class DataPoint:
    def __init__(self,dist=0,type=0):
        self.dist = dist
        self.type = type
    # 
    def __repr__(self):
        return repr(self.dist,self.type)

class HardCodedModel:
    def predict(data_test,k):
            pass
            
class HardCodedClassifier:
    def fit(data_train, targets_train):
        model = HardCodedModel
        return model
        
def main():
    
    auto_data = readData("/Users/parkerweech/Desktop/Education/Winter_2018/Machine_Learning_And_Data_Mining/auto.data")
    car_data = readData("/Users/parkerweech/Desktop/Education/Winter_2018/Machine_Learning_And_Data_Mining/car_data.data")
    pima_data = readData("/Users/parkerweech/Desktop/Education/Winter_2018/Machine_Learning_And_Data_Mining/pima_data.data")

    #This data set represents missing data with ?
    #Mixed
    auto_data = auto_data.replace(' ?', np.NaN)
    auto_data.dropna(inplace=True)
    
    #This data set doesn't appear to have missing data
    #All categorical
    car_data = car_data.replace(' ?', np.NaN)
    car_data.dropna(inplace=True)
  
    #This data set represents missing with 0, but 0 is also a target classifier  
    #All numerical
    auto_encoder = LabelEncoder()
    car_encoder = LabelEncoder()
    
    auto_data = encodeData(auto_data,auto_encoder)
    car_data = encodeData(car_data,car_encoder)    
    
    pima_x, pima_y = pima_data.iloc[:,:-1],pima_data.iloc[:,-1]
    car_x, car_y = car_data.iloc[:,:-1],car_data.iloc[:,-1]
    
    pima_x = pima_x.replace(' 0','1')
    
    pima_x = pima_x.values
    pima_y = pima_y.values
    car_x = car_x.values
    car_y = car_y.values
    
    
    pima_x_train, pima_x_test, pima_y_train, pima_y_test = train_test_split(
        pima_x, pima_y, test_size=0.2, random_state=0)
        
    car_x_train, car_x_test, car_y_train, car_y_test = train_test_split(
        car_x, car_y, test_size=0.2, random_state=0)
    
    print("Testing for pima data")
    kNN(pima_x_train, pima_x_test, pima_y_train, pima_y_test)
    
    print("Testing for car data")
    kNN(car_x_train, car_x_test, car_y_train, car_y_test)

    
if __name__ == "__main__":
    main()