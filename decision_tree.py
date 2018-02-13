import numpy as np
import pandas as pd
#from anytree import Node, RenderTree

# this is the classifier for the decision tree
class Decision_Tree_Classifier:
    pass
    
class Node:
    
    def __init__(self,name="",children={},leaf=False,num=0):
        self.name = name
        self.children = children
        self.leaf = leaf
        self.numChildren = num
            
    def __repr__(self):
        return "Feature: {}, Children: {}".format(self.name,self.children)
    
    def getName(self):
        return self.name
        
    def getNum(self):
        return self.numChildren
        
    def isLeaf(self):
        return self.leaf
    
    def addChild(self):
        self.numChildren += 1
        
    def setLeaf(self,isLeaf):
        self.leaf = isLeaf
        
    def getChildren(self):
        return self.children

# a function to read data info    
def read_data(link,col_names):
    data_set = pd.read_csv(link,header=None,names=col_names)
    return data_set
    
# a function to calc entropy of a given data_set
def calc_entropy(data):
    # calc -p(class)logbase2(p(class)) 
    p_data = data.value_counts()
    entropy = 0
    for index_1, val1 in p_data.iteritems():
        entropy -= (val1/len(data))*np.log2(val1/len(data))
        
    return entropy
        
# calculate the info gain
def calc_info_gain(ovr_ent,feature,data_set):
    # gain = ent - numFeature/total(entropy(feature)) - 
    # numFeature2/total(entropy(feature2)) ... 
    
    data = data_set[feature]
    labels = data_set['Class']
    gain = ovr_ent
    p_data = data.value_counts()
    l_data = labels.value_counts()
    
    count = 0
    # find out when one feature def matches which 
    for index_1, val1 in p_data.iteritems():
        ent = 0
        for index_2, val2 in l_data.iteritems():
            count = 0
            for index_3,row in data_set.iterrows():
                if index_1 == row[feature] and row['Class'] == index_2:
                    count += 1    
            if val1 > 0 and count > 0:
                ent -= (count/val1)*np.log2(count/val1)
        
        gain += (-(val1/len(data))*ent)       
    return gain
                          
# a function to build the tree recursively
def build_tree(dataset,num):
    # get entropy
    entropy = calc_entropy(dataset['Class'])

    # get info gain for every choice
    info_gains = {}
    for column in dataset:
        if column != 'Class':
            info_gains[column] = calc_info_gain(entropy,column,dataset)
    
    # make first selection
    node = Node(min(info_gains, key=info_gains.get))       
           
    # go left, make another selection
    # this is done by taking all the options that come from one option 
    # of that feature and making another node
    p_data = dataset[node.getName()].value_counts()
    
    for feature_val, val in p_data.iteritems():
        new_branch = pd.DataFrame()
        new_branch = dataset.copy()
        new_branch.drop(new_branch.index,inplace=True)
        print(feature_val)
      
        for index,row in dataset.iterrows():
            if(row[node.getName()] == feature_val):
                new_branch = new_branch.append(row)
        
        class_data = new_branch['Class'].value_counts()
                     
        if(len(new_branch.columns) <= 2 or class_data.nunique() == 1):
            leaf = pd.DataFrame()
            leaf = new_branch['Class'].mode().copy()
            leaf_node = Node(leaf[0])
            leaf_node.setLeaf(True)
            print("Leaf")   
            return leaf_node
            #print("This flower is a: ", leaf_node[0])
            
        elif(len(new_branch.columns) > 2):
            new_branch = new_branch.drop(node.getName(), axis=1)
            num += 1
            new_node = build_tree(new_branch,num)
            #print(new_node)
            node.children[feature_val] = new_node
            node.addChild()    
            
    return node 

# this function prints the tree
def print_tree(tree_node,num):
    print(tree_node)
    
    if tree_node.children != None:
         for child in tree_node.children:
             #if tree_node.children[child].isLeaf() == False:
                print(tree_node.children[child])

# main function
def main():
    col_names = ['Sepal_Length','Sepal_Width', 'Petal_Length','Petal_Width','Class']
    iris_data = read_data("/Users/parkerweech/Desktop/Education/Winter_2018/Machine_Learning_And_Data_Mining/iris.data.txt",
    col_names)
    
    # We're going to need to separate into train and test
    # This will possibly require some changes in our build_tree and possibly
    # Other functions..
    
    iris_data['Sepal_Length'] = pd.cut(iris_data['Sepal_Length'],3,labels=["small_0",
    "medium_0","large_0"])
    iris_data['Sepal_Width'] = pd.cut(iris_data['Sepal_Width'],3,labels=["small_1",
    "medium_1","large_1"])
    iris_data['Petal_Length'] = pd.cut(iris_data['Petal_Length'],3,labels=["small_2",
    "medium_2","large_2"])
    iris_data['Petal_Width'] = pd.cut(iris_data['Petal_Width'],3,labels=["small_3",
    "medium_3","large_3"])
    
    num = 0    
    tree_node = build_tree(iris_data,num)
    
    #print_tree(tree_node,num)
   
    # We need to implement the test
    
if __name__ == "__main__":
    main()