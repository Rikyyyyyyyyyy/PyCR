# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

data1 = [[1,2,3,5,2],[4,5,6,9,5],[7,8,9,2,3]]
data1 = np.array(data1)
y1 = [1,2,1]
valid_idx = [0,2]
data2 = data1[:,valid_idx]

clf = svm.SVC(kernel='linear', random_state=1, probability=False)
clf.fit(data1, y1)
class_pred = clf.predict(data1)
scu =  accuracy_score(y1, class_pred)

clf2 = svm.SVC(kernel='linear', random_state=1, probability=False)
clf2.fit(data2, y1)
class_pred2 = clf2.predict(data2)
scu2 =  accuracy_score(y1, class_pred2)
print(scu)
print(scu2)