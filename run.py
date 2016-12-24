import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

dataset=pd.read_csv("train.csv")
'''
pre = preprocessing.LabelEncoder()
pre.fit(dataset['skills'])
dataset['skills'] = pre.transform(dataset['skills'])
pre.fit(dataset['tag'])
dataset['tag'] = pre.transform(dataset['tag'])
'''
test=pd.read_csv("test.csv")
'''
pre.fit(test['skills'])
test['skills'] = pre.transform(test['skills'])
pre.fit(test['tag'])
test['tag'] = pre.transform(test['tag'])
'''
print dataset
print test
x=[]
y=[]
col=['user_id','problem_id','accuracy','error_count','solved_count_y','solved_count_x','attempts','level','rating','skills','tag','user_type']
x=dataset[col]
y=dataset['solved_status']
'''
bagging=BaggingClassifier(RandomForestClassifier(),max_samples=0.5,max_features=0.5)
bagging.fit(x,y)
predict=bagging.predict(test)
predict1=pd.DataFrame(predict)
predict1.to_csv("/home/flash/Pictures/india_hacks_ml/will_bill_solve_it/new_sub.csv")
'''


clf1 = AdaBoostClassifier(n_estimators = 1000)
clf1.fit(x,y)
predict1=clf1.predict(test)
predict1=pd.DataFrame(predict1)
#predict1.to_csv("ada.csv")
print "Model 1 "

clf2 = RandomForestClassifier(n_estimators=500,max_features=5,max_depth=None,min_samples_split=1,bootstrap=True,n_jobs=-1)
clf2.fit(x,y)
predict2=clf2.predict(test)
predict2=pd.DataFrame(predict2)
#predict2.to_csv("rf.csv")
print "Model 2 "


clf=VotingClassifier(estimators=[('ada', clf1), ('rf', clf2)], voting='hard')
clf.fit(x,y)
predict=clf.predict(test)
predict=pd.DataFrame(predict)
predict.to_csv("subcsv")


