import csv
from ForestFires import *
from scipy.io import loadmat
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from collections import Counter



spam = False
forest = False
kaggle = False

max_height = 7
num_trees = 70



def binarize(data): 
    return DictVectorizer().fit_transform(data).toarray()


def fill_in(data):
    averages = {}
    count = 0
    while count < len(data):
        row = data[count]
        for k,v in row.items():
            if count == 0:
                if isinstance(v, int):
                    averages[k] = v
                else:
                    averages[k] = [v]
            else:
                if v == '?':
                    continue  
                if isinstance(v, int):
                    averages[k] += v
                else:
                    averages[k].append(v)
        count += 1
    
    for k,v in averages.items():
        if isinstance(v,int):
            averages[k] = float(v)/len(data)  
        else:
            c = Counter(v)
            averages[k] = c.most_common(1)[0][0]
    for row in data:
        for k,v in row.iteritems():
            if v == '?':  
                row[k] = averages[k]
    return data


if spam:
    load = loadmat('spam_data/spam_data.mat')
    Xt = load['training_data']
    Yt = load['training_labels']
    Yt = Yt.ravel()
    Xv = load['test_data']
    if not kaggle:
        Xt, Xv, Yt, Yv = train_test_split(Xt, Yt, test_size=0.2)

else:
    df = read_csv('census_data/train_data.csv')
    df.drop('label', axis=1, inplace=True)
    reader = csv.DictReader(open('census_data/train_data.csv'))
    labels = []
    for row in reader:
        labels.append(int(row['label']))
    trainData = df.to_dict('records')
    Yt = np.asarray(labels)
    Xt = fill_in(trainData)
    Xt = binarize(Xt)
    df = read_csv('census_data/test_data.csv')
    reader = csv.DictReader(open('census_data/train_data.csv'))
    trainData = df.to_dict(orient = 'records')
    Xv = fill_in(trainData)
    Xv = binarize(Xv)
    if not kaggle:
        Xt, Xv, Yt, Yv = train_test_split(Xt, Yt, test_size=0.2)
    

if forest:
    classifier = RandomForest(num_trees=num_trees, max_height=max_height)
else:
    classifier = DecisionTree(max_height=max_height)


classifier.train(Xt, Yt)
Y_pred = classifier.predict(Xv)
Y_trainp = classifier.predict(Xt)



if kaggle:
    if spam:
        wrt = csv.writer(open('SpamPredictions.csv', 'wb'))
    
    if not spam:
        wrt = csv.writer(open('CensusPredictions.csv', 'wb'))
    
    wrt.writerow( ('Id', 'Category') )
    for i, label in enumerate(Y_pred):
        wrt.writerow( (int(i+1), int(label)) )

else:
    val_acc = accuracy_score(Yv, Y_pred)
    tr_acc = accuracy_score(Yt, Y_trainp)
    print( "Training Accuracy: %s" % tr_acc )
    print( "Validation Accuracy: %s" % val_acc )