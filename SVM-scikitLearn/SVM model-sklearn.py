# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.svm import SVC

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_recall_curve,average_precision_score,roc_curve,auc

from sklearn.preprocessing import label_binarize

import datetime
import time
import platform


inFile = '..\\..\\data\\data.csv'


train_set_rate = 0.8


pos_label = 0.0 
neg_label = 1.0 
pos_name = '坏用户'
neg_name = '好用户'

#parameters setup                                   
params = {'C': 10, 'kernel': 'rbf' , 'probability': True , 'class_weight':'auto'}      

#model setup
clf = SVC(**params)

params_prefix = 'rf_clf_' + str(params['C']) + '_' + str(params['kernel']) + '_' + str(params['probability']) + '_' + str(train_set_rate)
#file path setup
record_file = 'log/' + params_prefix+'_log.txt'
log_str = 'rf_clf: %s' % params + '\n'

sys_kind = platform.system()

if sys_kind == 'Windows':
    print_flag = True
else:
    print_flag = False
    
top_num = 20

#read data
inData = pd.read_csv(inFile)

#shuffle
x = inData.iloc[:,1:].values
y = inData.iloc[:,0:1].values

x,y = shuffle(x, y, random_state=13)

#mark
if pos_label == 1.0:
    y = label_binarize(y, classes=[0.0, 1.0])
else:
    y = label_binarize(y, classes=[1.0, 0.0])

y = list(y.reshape(y.size))


offset = int(len(x) * train_set_rate)

x_train , y_train = x[:offset], y[:offset]
x_test , y_test = x[offset:], y[offset:]


labelNames = np.array(inData.columns[1:])

#train the model
start_t = time.time()
cur_str = 'start training, %s' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
log_str += cur_str + '\n'
print cur_str

clf.fit(x_train, y_train)

cost_t = int(time.time() - start_t)
cur_str = 'end training, cost %d second' % cost_t
log_str += cur_str + '\n'
print cur_str


del x_train
del y_train

#prediction on test data
start_t = time.time()
cur_str = 'start predict, %s' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
log_str += cur_str + '\n'
print cur_str
    
y_predict = clf.predict(x_test)
y_scores = clf.predict_proba(x_test) 
y_scores = y_scores.T[1] 

cost_t = int(time.time() - start_t)
cur_str = 'end predict, cost %d second' % cost_t
log_str += cur_str + '\n'
print cur_str

# output:accuracy_score
accuracy = accuracy_score(y_test, y_predict)
cur_str = "\naccuracy_score: %.8f" % accuracy
log_str += cur_str + '\n'
if print_flag:
    print cur_str
    
#confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_predict)
tn = confusionMatrix[0][0]
fn = confusionMatrix[1][0]
tp = confusionMatrix[1][1]
fp = confusionMatrix[0][1]

try:
    precision_score = float(tp) / (tp + fp)
except ZeroDivisionError:
    precision_score = 0
    
try:
    recall_score = float(tp) / (tp + fn)
except ZeroDivisionError:
    recall_score = 0

try:
    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
except ZeroDivisionError:
    f1_score = 0
    
cur_str = "tn = %d, fn = %d, tp = %d, fp = %d\n" % (tn, fn, tp, fp)
cur_str += 'precision: %.2f, recall:%.2f, F1: %.2f\n' % (precision_score, recall_score, f1_score)
log_str += cur_str + '\n'
if print_flag:
    print cur_str

#classification_report
cur_str = classification_report(y_test, y_predict, labels=[0,1], target_names=[neg_name + '(neg)',pos_name + '(pos)'])    
log_str += cur_str + '\n'
if print_flag:
    print cur_str

#precision and recall rate
precision, recall, threshold = precision_recall_curve(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)

best_precision = 0
best_recall = 0
best_f1 = 0
best_th = 0
log_str += 'precision-recall pairs for different probability thresholds\n'
log_str += 'threshold,precision,recall,f1_score\n'
for i,th in enumerate(threshold):
    try:
        cur_f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    except ZeroDivisionError:
        cur_f1 = 0
    
    cur_str = '%.5f,%.2f,%.2f,%.2f' % (th,precision[i],recall[i],cur_f1)
    log_str += cur_str + '\n'
    
    if cur_f1 > best_f1:
        best_precision = precision[i]
        best_recall = recall[i]
        best_f1 = cur_f1
        best_th = th

cur_str = 'best thresholds: %.5f, precision: %.2f, recall: %.2f, F1_score: %.2f\n' % (best_th,best_precision,best_recall,best_f1)
log_str += cur_str + '\n'
if print_flag:
    print cur_str

#Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

#distribution function
def cal_score_distribution(y_test,y_predict,pos_label=1,neg_label=0,gap_array=None,outFile=None):
    result = list()
    
    check = pd.DataFrame({'test':y_test, 'predict':y_predict})
    if gap_array is None:
        gap_array = np.arange(0,1,0.01)
    
    pos_total = (check.test == pos_label).value_counts()[True]
    for i in range(len(gap_array) - 1):
        try:
            cnt_pos = ((check.predict >= gap_array[i]) & (check.predict < gap_array[i + 1]) & (check.test == pos_label)).value_counts()[True]
        except KeyError:
            cnt_pos = 0
            
        try:
            cnt_neg = ((check.predict >= gap_array[i]) & (check.predict < gap_array[i + 1]) & (check.test == neg_label)).value_counts()[True]
        except KeyError:
            cnt_neg = 0
        
        try:
            pos_inner_cent = float(cnt_pos)/(cnt_pos+cnt_neg)
        except ZeroDivisionError:
            pos_inner_cent = 0
            
        try:
            pos_cent = float(cnt_pos)/(pos_total)
        except ZeroDivisionError:
            pos_cent = 0
        
        result.append(['%.2f - %.2f' % (gap_array[i], gap_array[i + 1]),cnt_pos + cnt_neg,cnt_pos,cnt_neg,pos_inner_cent,pos_cent])
    
    #save results
    if outFile is not None:
        with open(outFile,'w') as f:
            for score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent in result:
                f.write('%s,%d,%d,%d,%.2f,%.2f\n' % (score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent))
    
    return result

#distribution for prediction
gap_array = np.arange(0,1.0001,0.05)
distribution_result = cal_score_distribution(y_test,y_scores,pos_label=1,neg_label=0,gap_array=gap_array)
log_str += 'score_distribution\n'
log_str += 'score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent\n'
for score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent in distribution_result:
    log_str += '%s,%d,%d,%d,%.2f,%.2f\n' % (score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent)

if sys_kind == 'Windows':
    cur_df = pd.DataFrame(distribution_result)
    cur_df.columns = ['score_arange','cnt_total','cnt_pos','cnt_neg','pos_inner_cent','pos_cent']
    cur_df.set_index('score_arange')
    cur_df = cur_df[['pos_inner_cent','pos_cent']]
    cur_df.plot(kind='bar',title='score distribution',subplots=True)

# weights for features
if params['kernel'] == 'linear':
    feature_importance = clf.coef_[0]
    #conversion to relative features
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    
    a = feature_importance[sorted_idx]
    b = labelNames[sorted_idx]
    log_str += '\nfeature importances\n'
    log_str += 'feature_name,weight\n'
    with open(record_file,'w') as f:
        f.write(log_str)
        for i,weight in enumerate(a):
            if weight > 0:
                f.write(str(b[i]) + ',' + str(weight) + '\n')


#visualization
if sys_kind == 'Windows':
    import matplotlib.pyplot as plt
    
    #plot Precision-Recall curve
    pr_fig = plt.figure('Precision-Recall curve')
    pr_sub = pr_fig.add_subplot(1,1,1)
    pr_sub.plot(recall, precision, label='Precision-Recall curve')
    pr_sub.set_xlabel('Recall')
    pr_sub.set_ylabel('Precision')
    pr_sub.set_ylim([0.0, 1.05])
    pr_sub.set_xlim([0.0, 1.0])
    pr_sub.set_title('Precision-Recall: area=%0.2f th=%0.2f p=%0.2f r=%0.2f f1=%0.2f' % (average_precision,best_th,best_precision,best_recall,best_f1))
    pr_sub.legend(loc="lower left")
    
    #ROC curve
    plt.figure('Receiver operating characteristic curve')
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    #weights for the top n features
    if params['kernel'] == 'linear':
        weight_fig = plt.figure('Variable Importance')
        weight_sub = weight_fig.add_subplot(1,1,1)
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        weight_sub.barh(pos[(0 - top_num):], feature_importance[sorted_idx][(0 - top_num):], align='center')
        weight_sub.set_yticks(pos[(0 - top_num):])
        weight_sub.set_yticklabels(labelNames[sorted_idx][(0 - top_num):])
        weight_sub.set_xlabel('Relative Importance')
        weight_sub.set_title('Variable Importance')
    
    
    plt.show()

