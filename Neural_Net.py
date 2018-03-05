import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random as rd

class Node: 
    def __init__(self,layer=0,num=0,h=0,g=0,d=0,o=-1,intended=-1):
        self.layer = layer
        self.num = num
        self.h = h
        self.g = g
        self.d = d
        self.o = o
        self.intended = intended
        
    def __repr__(self):
        return "Layer: {} Num: {}".format(self.layer,self.num)
    
    def update(self,h):
        self.h += h
        self.g = (1/(1+np.e**(-1*self.h)))
        
    def getG(self):
        return self.g
        
    def setO(self,o):
        self.o = o
        
    def getO(self):
        return self.o
        
    def setIntended(self,intended):
        self.intended = intended
        
    def getIntended(self):
        return self.intended
    
    def getH(self):
        return self.h
    
    def setD(self,d):
        self.d = d  
        
    def getD(self):
        return self.d
    
    def clearNode(self):
        self.h = 0
        self.g = 0
        self.d = 0             
        
def read_data(link,col_names):
    data_set = pd.read_csv(link,header=None,names=col_names)
    return data_set
    
def setupWeights(layers):
    i = 0
    weight_list = []
    
    for layer in layers:
        if i == 0:
            #handle input
            weights = []
            for x in range(0,layers[i]):
                for y in range(0,layers[i+1]):
                     # here we need to create weights for each connection of
                     # nodes. These must be stored thru the entire forward and backward prop   
                    weight = rd.uniform(-1,1)
                    weights.append(weight)
                    
            weight_list.append(weights)
            i+=1
            
        elif i == (len(layers)-1):
            #handle output
            
            i+=1
        else:
            #handle hidden
            weights = []
            for x in range(0,layers[i]):
                for y in range(0,layers[i+1]):
                     # here we need to create weights for each connection of
                     # nodes. These must be stored thru the entire forward and backward prop   
                    weight = rd.uniform(-1,1)
                    weights.append(weight)
                    
            weight_list.append(weights)
            
            i+=1
            
    return(weight_list)
    
def createLayers(data,targets):
    num_h = int(input("How many hidden layers would you like to use? "))
    # bias = int(input("Would you like to include a bias node? (0 = no, !0 = yes): "))
    num_l = num_h + 2
    bias = 1
    
    if bias == 0:
        b_bias = False
    elif bias != 0:
        b_bias = True
    
    layers = []
    
    for i in range(0,num_l):
        if i == 0:
            if b_bias == False:
                layers.append(len(data.columns))
            else:
                layers.append(len(data.columns)+1)      
        elif i < num_l - 1:
            num_n = int(input("How many nodes would you like to use in this hidden layer? "))
            if b_bias == False:
                layers.append(num_n)
            else:
                layers.append(num_n+1)
        else:
            layers.append(targets.nunique())
  
    return layers,b_bias      
    
def clearNodeListValues(node_list):
    for node_layer in node_list:
        for node in node_layer:
            node.clearNode()
            
def computeError(weight_list,layers,node_list,lr):
    
    n_weight_list = []
    
    for weight_layer in weight_list:
        n_weight_layer = []
        for weight in weight_layer:
            n_weight_layer.append(0)
        n_weight_list.append(n_weight_layer)

    for x in range((len(layers)-1),-1,-1):
        if x == len(layers)-1:
            # handle output layer error
            # error = output value(0,1) * (1 - outputvalue) * (outputvalue - intended)
            # weight - learning rate * (output - intended output
            # y is the node in the current layer
            for y in range(0,layers[x]):
                a = node_list[x][y].getG()
                i = node_list[x][y].getIntended()
                d = (a-i)*(a)*(1-a)
                node_list[x][y].setD(d)
                
                # z is the node in the previous layer
                for z in range(0,layers[x-1]):
                    num = (z*layers[x]+y)
                    n_weight_list[x-1][num] = (weight_list[x-1][num]) - (lr*d*(node_list[x-1][z].getG()))
            
        elif x == 0:
            # this is the input layer. 
            # no error to be calculated here
           pass
            
        else:
            # handle hidden layer error
             # remember, y is nodes in the current layer!
            for y in range(0,layers[x]):
                a = node_list[x][y].getG()
                d1 = a*(1-a)
                
                sum = 0
                # and we're going to add k to be the nodes in the next layer
                for k in range(0,layers[x+1]):
                    num = (y*layers[x+1]+k)
                    sum += ((weight_list[x][num])*(node_list[x+1][k].getD()))
                
                d2 = d1 * sum
                node_list[x][y].setD(d2)
                 
                # we're going to keep z nodes in the previous layer
                for z in range(0,layers[x-1]):
                    num = (z*layers[x]+y)
                    n_weight_list[x-1][num] = (weight_list[x-1][num]) - (lr*d1*(node_list[x-1][z].getG()))                
    
    return(n_weight_list)
            
    
def runNet(weight_list,layers,df,targets,bias,node_list):
    
    # num_e = int(input("How many epochs would you like to run? "))
    # learn_rate = float(input("What learning rate would you like to use? "))
    
    num_e = 500
    learn_rate = .87

    data = df.values
    target = targets.values
    target_values = np.unique(target)
    
    for epoch in range(0,num_e):
        if epoch % 25 == 0 and epoch != 0:
            print(epoch)
            
        j = 0
        correct = 0
        for instance in data:
            
            clearNodeListValues(node_list)
            
            if bias == True:
                instance = np.insert(instance,0,-1,axis=0)
                
            # i represents the current layer!
            i = 0
            # layer is simply a number representing how many nodes/inputs are in this layer
            for layer in layers:
                if i == 0:
                    #handle input
                    # x represents the current node in the current layer
                    for x in range(0,layers[i]):
                        # y represents the current node in the next layer
                        # therefore the connection between x and y gives us the connections of nodes
                        for y in range(0,layers[i+1]):
                            value = weight_list[i][(x*layers[i+1])+y]*instance[x]
                            node_list[i+1][y].update(value)
                            #print(node_list[i+1][y].getG())
                      
                    i+=1
                    
                elif i == (len(layers)-1):
                    for x in range(0,layers[i]):
                        #print(node_list[i][x].getG())
                        if node_list[i][x].getG() >= .5:
                            node_list[i][x].setO(1)
                            #print("Activated")
                        elif node_list[i][x].getG() < .5:
                            node_list[i][x].setO(0)
                            #print("Not activated")
                            
                        if target[j] == target_values[x]:
                            node_list[i][x].setIntended(1)
                            #print(target[j],"==", target_values[x])
                        elif target[j] != target_values[x]:
                            node_list[i][x].setIntended(0)
                            #print(target[j],"!=", target_values[x])   
                    
                    # let's check our accuracy   
                    right = 1  
                    for x in range(0,layers[i]):
                        if node_list[i][x].getO() == node_list[i][x].getIntended():
                            #print(node_list[i][x].getO(), "==", node_list[i][x].getIntended())
                            pass
                        else:
                            #print(node_list[i][x].getO(), "!=", node_list[i][x].getIntended())
                            right = 0
                        
                    if right == 1:
                       correct += 1           
                            
                else:
                    #handle hidden
                    for x in range(0,layers[i]):
                        for y in range(0,layers[i+1]):
                            if x == 0:
                                value = weight_list[i][(x*layers[i+1])+y]*-1
                                node_list[i+1][y].update(value)
                                #print(value)
                                #print(node_list[i+1][y].getG())
                            else:
                                value = weight_list[i][(x*layers[i+1])+y]*node_list[i][x].getG()
                                node_list[i+1][y].update(value)
                                #print(value)
                                #print(node_list[i+1][y].getG())
                    
                    i+=1 
            weight_list = computeError(weight_list,layers,node_list,learn_rate) 
            j+=1 
            
        print("On epoch #{} we got {} correct out of {} total".format(epoch,correct,j))
    return weight_list
    
def createNodeList(layers):
    i = 0
    node_list = []
    
    for layer in layers:
        node_layer = []
        
        for x in range(0,layer):
            node = Node(layer=i,num=x)
            node_layer.append(node)
            
        node_list.append(node_layer)
        i += 1 
           
    return node_list
        
def main():
    # Lets get the dataset 
    
    col_names = ['Sepal_Length','Sepal_Width', 'Petal_Length','Petal_Width','Class']
    iris_data = read_data("/Users/parkerweech/Desktop/Education/Winter_2018/Machine_Learning_And_Data_Mining/iris.data.txt",
    col_names)
    
    iris_data, iris_targets = iris_data.iloc[:,:-1],iris_data.iloc[:,-1]
    iris_x_train, iris_y_train, iris_x_test, iris_y_test = train_test_split(
        iris_data, iris_targets, test_size=0.2, random_state=0)
    
    layers,bias = createLayers(iris_x_train,iris_x_test)
    node_list = createNodeList(layers)
    weight_list = setupWeights(layers)
    final_weights = runNet(weight_list,layers,iris_x_train,iris_x_test,bias,node_list)
        
if __name__ == "__main__":
    main()