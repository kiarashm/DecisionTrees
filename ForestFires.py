import numpy as np
from math import ceil
from collections import Counter
from random import randint


class Node():
    def __init__(self, split=None, left=None, right=None, label=None):
        self.split = split
        self.left = left
        self.right = right
        self.label = label

    def isLeaf(self):
        if self.label == None:
            return False
        return True


class DecisionTree():
    def __init__(self, max_height=None):
        self.root = None
        self.max_height = max_height


    def grow(self, data, labels, curr_depth, max_depth):
        if np.unique(labels).size == 1:
            return Node(label=labels[0])

        feature, threshold = self.segmentor(data, labels)

        if curr_depth == max_depth or not feature:
            c = Counter(labels)
            mode = c.most_common(1)[0][0]
            return Node(label=mode)

        cond = (data[:,feature] < threshold)
        X_l = data[cond]
        Y_l = labels[cond]
        X_r = data[~cond]
        Y_r = labels[~cond]  

        leftTree = self.grow(X_l, Y_l, curr_depth+1, max_depth)
        rightTree = self.grow(X_r, Y_r, curr_depth+1, max_depth)

        return Node(split=(feature, threshold), left=leftTree, right=rightTree)

    def impurity(self, Y_l, Y_r):
        H_l = self.entropy(Y_l)
        H_r = self.entropy(Y_r)
        return float((len(Y_l) * H_l) + (len(Y_r) * H_r)) / float(len(Y_l) + len(Y_r))

    def entropy(self, Y):
        p_1 = float(Y.sum()) / float(len(Y))
        p_0 = 1. - p_1

        if p_0 == 0:
            h_0 = 0 
        if p_1 == 0:
            h_1 = 0 
        if p_0 != 0:
            h_0 = -p_0 * np.log2(p_0)
        if p_1 != 0:
            h_1 = -p_1 * np.log2(p_1)
        
        return h_0 + h_1


    def compress_features(self, vec):
        uniq = np.unique(vec)
        if uniq.size > 1000:
            rnd = lambda x: int(ceil(x / 10000.0))*10000
            return np.unique(map(rnd, uniq))
        elif np.array_equal(uniq, [0,1]):
            return [1]
        return uniq

    def segmentor(self, data, labels):
        best_feat = None 
        best_threshold = None
        best_impurity = float('inf')

        for i in range(len(data.T)):
            feat_vec = data.T[i]

            for threshold in self.compress_features(feat_vec):
                cond = (data[:,i] < threshold)
                Y_l = labels[cond]
                Y_r = labels[~cond]  

                if len(Y_l) == 0 or len(Y_r) == 0:  
                    continue

                curr_impurity = self.impurity(Y_l, Y_r)

                if curr_impurity < best_impurity:
                    best_feat, best_threshold = i, threshold
                    best_impurity = curr_impurity

        return (best_feat, best_threshold)


    def train(self, data, labels):
        self.root = self.grow(data, labels, 0, self.max_height)

    def predict(self, data): 
        predictions = []
        for i in range(len(data)):
            x = data[i]
            curr = self.root
            while curr.isLeaf() == False:
                feat, thresh = curr.split
                if (i == 1):
                    print(feat, thresh)
                if (x[feat] >= thresh):
                    curr = curr.right
                else:
                    curr = curr.left
        
            predictions.append(curr.label)


        return np.asarray(predictions).ravel()



class RandomForest():
    def __init__(self, num_trees, max_height=None):
        self.num_trees = num_trees
        self.max_height = max_height
        trees = []
        for x in range(num_trees):
            trees.append(DecisionTree(max_height=max_height))
        self.trees = trees

    def train(self, data, labels):
        numSamples = data.shape[0] / self.num_trees
        for i in range(self.num_trees):
            rand = np.random.choice(data.shape[0], numSamples)
            self.trees[i].train(data[rand], labels[rand])

    def predict(self, data):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(data))
        avg_preds = np.average(predictions, axis=0)
        for i in range(len(avg_preds)):
            if avg_preds[i] >= 0.5:
                avg_preds[i] = 1
            else:
                avg_preds[i] = 0
        return avg_preds

