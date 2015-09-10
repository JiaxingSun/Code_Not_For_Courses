# -*- coding: UTF-8 -*- 


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import sys

class UserCreditPredict:
    def __init__(self):
        

        self.usedColumns = [u"入网时间",u"套餐价格",u"每月流量",u"每月话费",u"每月通话时长",u"欠费金额",u"欠费月份数"]
        self.data_mean = np.asarray([38.24266732,112.6369674,177.4180309,79.5166082,260.6016648,30.29175989,0.447792346])
        self.data_std = np.asarray([20.79561656,112.5224782,460.1787543,92.29306833,304.0502764,109.6174008,0.943458343])
        self.pca1_coef = np.asarray([0.267920529,-0.445496651,-0.431920331,-0.556188417,-0.369916786,-0.269068409,-0.15688964])
        self.pca2_coef = np.asarray([0.030523999,0.143306846,0.193534745,0.106744508,0.219601865,-0.639079362,-0.687774626])
        self.pca3_coef = np.asarray([0.730366217,-0.382786266,0.079605108,0.189963042,0.51229872,0.083454306,0.090566732])

        self.modelSavePath = 'model/decisionTree.pkl'
        self.traingDataPath = 'training_data.csv'
        self.load(self.modelSavePath)
        
    def load(self,modelSavePath):
        self.model = joblib.load(modelSavePath)
        print 'model successfully loaded from %s' % modelSavePath
    
   
    def train(self,trainingDataPath,modelSavePath):
        traing_data_full = pd.read_csv(self.traingDataPath,encoding='gbk')
        traing_data_X = traing_data_full[['pca_1','pca_2','pca_3']]
        traing_data_Y = traing_data_full['rating_label']
        model = tree.DecisionTreeClassifier(max_depth=4)
        model.fit(traing_data_X, traing_data_Y)
        self.model = model
        joblib.dump(model, self.modelSavePath) 
        print 'Training done! model saved to file %s' % modelSavePath
    
    
    def predict(self, testDataPath, resultSavePath):
        #if self.model == None:
            #self.train(self.traingDataPath,self.modelSavePath)
        testData = None
        if testDataPath.endswith(".csv"):
            testData = pd.read_csv(testDataPath,encoding='gbk')
        elif ( testDataPath.endswith(".xls") or testDataPath.endswith(".xlsx")):
            testData = pd.read_excel(testDataPath,encoding='gbk')
       
        testDataIDRemoved = testData[self.usedColumns]
       
        test_data_scaled = (testDataIDRemoved - self.data_mean)/self.data_std
        
        test_pca1 = test_data_scaled.dot(self.pca1_coef)
        test_pca2 = test_data_scaled.dot(self.pca2_coef)
        test_pca3 = test_data_scaled.dot(self.pca3_coef)

        merged_test_pca = pd.concat([test_pca1, test_pca2, test_pca3],axis=1)
        merged_test_pca.columns = ['pca_1','pca_2','pca_3']
        y_test_pred = self.model.predict(merged_test_pca)
        y_test_pred_frame = pd.DataFrame(y_test_pred,columns=['predicted_credit'])
        result = pd.concat([testData, y_test_pred_frame],axis=1)
        
        
        if resultSavePath.endswith(".csv"):
            result.to_csv(resultSavePath,encoding='gbk',index=None)
        elif ( resultSavePath.endswith(".xls") or resultSavePath.endswith(".xlsx")):
            result.to_excel(resultSavePath,encoding='gbk',index=None)
        print 'done! results saved to file %s' % resultSavePath
        
def main(argv):
    testDataPath = argv[1]
    saveDataPath = argv[2]
    print testDataPath
    print saveDataPath
    ratingModel = UserCreditPredict()
    ratingModel.predict(testDataPath, saveDataPath)
        
if __name__ == '__main__':
    main(sys.argv)



    