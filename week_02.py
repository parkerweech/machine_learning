from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from numpy import sqrt


iris = datasets.load_iris()
        
class DataPoint:
    def __init__(self,dist=0,type=0):
        self.dist = dist
        self.type = type
    
    def __repr__(self):
        return repr(self.dist,self.type)
        
def main():

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0)

    #print(X_train)
    #print(y_train)
    
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
            dist = sqrt((X_test[i][0] - X_train[j][0])**2 + 
                        (X_test[i][1] - X_train[j][1])**2 + 
                        (X_test[i][2] - X_train[j][2])**2 + 
                        (X_test[i][3] - X_train[j][3])**2)
            
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
    
#    print("Now here is a side by side comparison of both lists:")
    
#    index_2 = 0
#    for y in targets_predicted:
#        print("Ours: ",targets_predicted[index_2],". Theirs: ", predictions[index_2])
#        index_2 += 1
    
if __name__ == "__main__":
    main()