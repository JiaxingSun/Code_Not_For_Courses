# -*- coding: UTF-8 -*- 


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import sys

class UserCreditTrain:
    def __init__(self):
        self.modelSavePath = 'model/decisionTree.pkl'
        self.traingDataPath = 'training_data.csv'

    def train(self,trainingDataPath):
        traing_data_full = pd.read_csv(trainingDataPath,encoding='gbk')
        traing_data_X = traing_data_full[['pca_1','pca_2','pca_3']]
        traing_data_Y = traing_data_full['rating_label']
        model = tree.DecisionTreeClassifier(max_depth=4)
        model.fit(traing_data_X, traing_data_Y)
        self.model = model
        joblib.dump(model, self.modelSavePath) 
        print 'Training done! model saved to file %s' % self.modelSavePath
        
def main(argv):
    testDataPath = argv[1]
    print testDataPath
    ratingModel = UserCreditTrain()
    ratingModel.train(testDataPath)
        
if __name__ == '__main__':
    main(sys.argv)



    