#Project name: Implementation of k-NN Classifier
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#function for loading data

import numpy as np

batch1 = unpickle("data_batch_1")
#loading data. It will be the first dataset of 5fold cross validation.
Xtr1 = batch1[b'data']
#key of train data
Ytr1 = np.array(batch1[b'labels'])
#key of train labels

batch2 = unpickle("data_batch_2")
#loading second data
Xtr2 = batch2[b'data']
Ytr2 = np.array(batch2[b'labels'])

batch3 = unpickle("data_batch_3")
#loading third data
Xtr3 = batch3[b'data']
Ytr3 = np.array(batch3[b'labels'])

batch4 = unpickle("data_batch_4")
#loading 4th data
Xtr4 = batch4[b'data']
Ytr4 = np.array(batch4[b'labels'])

batch5 = unpickle("data_batch_5")
#loading 5th data
Xtr5 = batch5[b'data']
Ytr5 = np.array(batch5[b'labels'])
#loading the rest of dataset

testbatch = unpickle("test_batch")
#loading test data
Xte = testbatch[b'data']
Yte = np.array(testbatch[b'labels'])
Xtr = np.append(np.append(np.append(Xtr1, Xtr2, axis=0), np.append(Xtr3, Xtr4, axis=0), axis=0), Xtr5, axis=0)
#append full train data
Ytr = np.append(np.append(np.append(Ytr1, Ytr2, axis=0), np.append(Ytr3, Ytr4, axis=0), axis=0), Ytr5, axis=0)
#append full train labels

class NearestNeighbor:
    def __init__(self):
#initiator
        pass
    def train(self, X, y):
#add train function
        self.Xtr = X
#assign train data
        self.ytr = y
#assign train labels

    def cval_5f_L1(self, f1_x, f2_x, f3_x, f4_x, f5_x, f1_y, f2_y, f3_y, f4_y, f5_y, k_index):
#add 5fold cross validation function for L1 metric
        print('Execute a 5fold cross validation (L1 norm)')
        f_num = f1_x.shape[0]
#f_num is length of a fold data. All fold data has same length
        edge12 = np.zeros((f_num, f_num))
#connect first and second fold data.
        edge13 = np.zeros((f_num, f_num))
#connect 1st and 3rd
        edge14 = np.zeros((f_num, f_num))
#connect 1st and 4th
        edge15 = np.zeros((f_num, f_num))
#connect 1st and 5th
        edge23 = np.zeros((f_num, f_num))
#connect 2nd and 3rd
        edge24 = np.zeros((f_num, f_num))
#connect 2nd and 4th
        edge25 = np.zeros((f_num, f_num))
#connect 2nd and 5th
        edge34 = np.zeros((f_num, f_num))
#connect 3rd and 4th
        edge35 = np.zeros((f_num, f_num))
#connect 3rd and 5th
        edge45 = np.zeros((f_num, f_num))
#connect 4th and 5th

        for i in range(f_num):
#calculate edge weight between two nodes. The weight will be L1 distance
            edge12[:, i] = np.sum(np.abs(f1_x - f2_x[i, :]), axis = 1)
#between node1 and node2
            edge13[:, i] = np.sum(np.abs(f1_x - f3_x[i, :]), axis = 1)
#between node1 and node3
            edge14[:, i] = np.sum(np.abs(f1_x - f4_x[i, :]), axis = 1)
#between node1 and node4
            edge15[:, i] = np.sum(np.abs(f1_x - f5_x[i, :]), axis = 1)
#between node1 and node5
            edge23[:, i] = np.sum(np.abs(f2_x - f3_x[i, :]), axis = 1)
#between node2 and node3
            edge24[:, i] = np.sum(np.abs(f2_x - f4_x[i, :]), axis = 1)
#between node2 and node4
            edge25[:, i] = np.sum(np.abs(f2_x - f5_x[i, :]), axis = 1)
#between node2 and node5
            edge34[:, i] = np.sum(np.abs(f3_x - f4_x[i, :]), axis = 1)
#between node3 and node4
            edge35[:, i] = np.sum(np.abs(f3_x - f5_x[i, :]), axis = 1)
#between node3 and node5
            edge45[:, i] = np.sum(np.abs(f4_x - f5_x[i, :]), axis = 1)
#between node4 and node5

        edge21 = edge12.T
#between node2 and node1. It is exactly same as transpose of edge12
        edge31 = edge13.T
#between node3 and node1
        edge41 = edge14.T
#between node4 and node1
        edge51 = edge15.T
#between node5 and node1
        edge32 = edge23.T
#between node3 and node2
        edge42 = edge24.T
#between node4 and node2
        edge52 = edge25.T
#between node5 and node2
        edge43 = edge34.T
#between node4 and node3
        edge53 = edge35.T
#between node5 and node3
        edge54 = edge45.T
#between node5 and node4

        print('Edges were successfully created')
        acc = np.zeros(k_index.shape[0])
        val1 = np.append(np.append(edge21, edge31, axis=0), np.append(edge41, edge51, axis=0), axis=0)
#make first dataset of 5fold cross validation
        val1_l = np.append(np.append(f2_y, f3_y), np.append(f4_y, f5_y))
        val1_s = np.sort(val1, axis=0)
        val1_as = np.argsort(val1, axis=0)
        val2 = np.append(np.append(edge12, edge32, axis=0), np.append(edge42, edge52, axis=0), axis=0)
#2nd dataset
        val2_l = np.append(np.append(f1_y, f3_y), np.append(f4_y, f5_y))
        val2_s = np.sort(val2, axis=0)
        val2_as = np.argsort(val2, axis=0)
        val3 = np.append(np.append(edge13, edge23, axis=0), np.append(edge43, edge53, axis=0), axis=0)
#3rd dataset
        val3_l = np.append(np.append(f1_y, f2_y), np.append(f4_y, f5_y))
        val3_s = np.sort(val3, axis=0)
        val3_as = np.argsort(val3, axis=0)
        val4 = np.append(np.append(edge14, edge24, axis=0), np.append(edge34, edge54, axis=0), axis=0)
#4th dataset
        val4_l = np.append(np.append(f1_y, f2_y), np.append(f3_y, f5_y))
        val4_s = np.sort(val4, axis=0)
        val4_as = np.argsort(val4, axis=0)
        val5 = np.append(np.append(edge15, edge25, axis=0), np.append(edge35, edge45, axis=0), axis=0)
#5th dataset
        val5_l = np.append(np.append(f1_y, f2_y), np.append(f3_y, f4_y))
        val5_s = np.sort(val5, axis=0)
        val5_as = np.argsort(val5, axis=0)
        for k in k_index:
            
#start voting
            val1_index = np.zeros((f_num, 10))
            val2_index = np.zeros((f_num, 10))
            val3_index = np.zeros((f_num, 10))
            val4_index = np.zeros((f_num, 10))
            val5_index = np.zeros((f_num, 10))
            
            for p in range(k):
#take k'th data in sorting dataset
                for j in range(f_num):
#input weight which is number one divided by distance in result list
                    val1_index[j, val1_l[val1_as[p, j]]] = val1_index[j, val1_l[val1_as[p, j]]] + 1 / (val1_s[p, j] + 0.001)
#for val1. 0.001 is for zero distance
                    val2_index[j, val2_l[val2_as[p, j]]] = val2_index[j, val2_l[val2_as[p, j]]] + 1 / (val2_s[p, j] + 0.001)
#for val2
                    val3_index[j, val3_l[val3_as[p, j]]] = val3_index[j, val3_l[val3_as[p, j]]] + 1 / (val3_s[p, j] + 0.001)
#for val3
                    val4_index[j, val4_l[val4_as[p, j]]] = val4_index[j, val4_l[val4_as[p, j]]] + 1 / (val4_s[p, j] + 0.001)
#for val4
                    val5_index[j, val5_l[val5_as[p, j]]] = val5_index[j, val5_l[val5_as[p, j]]] + 1 / (val5_s[p, j] + 0.001)
#for val5
            acc[k_index == k*np.ones(k_index.shape[0])] = np.mean(
                [np.mean(f1_y.T == np.argmax(val1_index, axis = 1)), np.mean(f2_y.T == np.argmax(val2_index, axis = 1)),
                 np.mean(f3_y.T == np.argmax(val3_index, axis = 1)), np.mean(f4_y.T == np.argmax(val4_index, axis = 1)),
                 np.mean(f5_y.T == np.argmax(val5_index, axis = 1))])
    
#calculate accuracy
        print('5fold cross validation was successfully completed (L1 norm)')
        return acc
    
#5fold cross validation function for L2 metric is exactly same as L1 metric except weighting edges.
#So, I will skip comments in this part, but you can see detail concept in 'report.docx'
    def cval_5f_L2(self, f1_x, f2_x, f3_x, f4_x, f5_x, f1_y, f2_y, f3_y, f4_y, f5_y, k_index):
        print('Execute a 5fold cross validation (L2 norm)')
        f_num = f1_x.shape[0]
        edge12 = np.zeros((f_num, f_num))
        edge13 = np.zeros((f_num, f_num))
        edge14 = np.zeros((f_num, f_num))
        edge15 = np.zeros((f_num, f_num))
        edge23 = np.zeros((f_num, f_num))
        edge24 = np.zeros((f_num, f_num))
        edge25 = np.zeros((f_num, f_num))
        edge34 = np.zeros((f_num, f_num))
        edge35 = np.zeros((f_num, f_num))
        edge45 = np.zeros((f_num, f_num))
        for i in range(f_num):
            edge12[:, i] = np.sqrt(np.sum(np.square(f1_x - f2_x[i, :]), axis=1))
#weighting edge in L2 distance
            edge13[:, i] = np.sqrt(np.sum(np.square(f1_x - f3_x[i, :]), axis=1))
            edge14[:, i] = np.sqrt(np.sum(np.square(f1_x - f4_x[i, :]), axis=1))
            edge15[:, i] = np.sqrt(np.sum(np.square(f1_x - f5_x[i, :]), axis=1))
            edge23[:, i] = np.sqrt(np.sum(np.square(f2_x - f3_x[i, :]), axis=1))
            edge24[:, i] = np.sqrt(np.sum(np.square(f2_x - f4_x[i, :]), axis=1))
            edge25[:, i] = np.sqrt(np.sum(np.square(f2_x - f5_x[i, :]), axis=1))
            edge34[:, i] = np.sqrt(np.sum(np.square(f3_x - f4_x[i, :]), axis=1))
            edge35[:, i] = np.sqrt(np.sum(np.square(f3_x - f5_x[i, :]), axis=1))
            edge45[:, i] = np.sqrt(np.sum(np.square(f4_x - f5_x[i, :]), axis=1))
        edge21 = edge12.T
        edge31 = edge13.T
        edge41 = edge14.T
        edge51 = edge15.T
        edge32 = edge23.T
        edge42 = edge24.T
        edge52 = edge25.T
        edge43 = edge34.T
        edge53 = edge35.T
        edge54 = edge45.T
        print('Edges were successfully created')
        acc = np.zeros(k_index.shape[0])
        val1 = np.append(np.append(edge21, edge31, axis=0), np.append(edge41, edge51, axis=0), axis=0)
        val1_l = np.append(np.append(f2_y, f3_y), np.append(f4_y, f5_y))
        val1_s = np.sort(val1, axis=0)
        val1_as = np.argsort(val1, axis=0)
        val2 = np.append(np.append(edge12, edge32, axis=0), np.append(edge42, edge52, axis=0), axis=0)
        val2_l = np.append(np.append(f1_y, f3_y), np.append(f4_y, f5_y))
        val2_s = np.sort(val2, axis=0)
        val2_as = np.argsort(val2, axis=0)
        val3 = np.append(np.append(edge13, edge23, axis=0), np.append(edge43, edge53, axis=0), axis=0)
        val3_l = np.append(np.append(f1_y, f2_y), np.append(f4_y, f5_y))
        val3_s = np.sort(val3, axis=0)
        val3_as = np.argsort(val3, axis=0)
        val4 = np.append(np.append(edge14, edge24, axis=0), np.append(edge34, edge54, axis=0), axis=0)
        val4_l = np.append(np.append(f1_y, f2_y), np.append(f3_y, f5_y))
        val4_s = np.sort(val4, axis=0)
        val4_as = np.argsort(val4, axis=0)
        val5 = np.append(np.append(edge15, edge25, axis=0), np.append(edge35, edge45, axis=0), axis=0)
        val5_l = np.append(np.append(f1_y, f2_y), np.append(f3_y, f4_y))
        val5_s = np.sort(val5, axis=0)
        val5_as = np.argsort(val5, axis=0)
        for k in k_index:
            val1_index = np.zeros((f_num, 10))
            val2_index = np.zeros((f_num, 10))
            val3_index = np.zeros((f_num, 10))
            val4_index = np.zeros((f_num, 10))
            val5_index = np.zeros((f_num, 10))
            for p in range(k):
                for j in range(f_num):
                    val1_index[j, val1_l[val1_as[p, j]]] = val1_index[j, val1_l[val1_as[p, j]]] + 1 / (val1_s[p, j] + 0.001)
                    val2_index[j, val2_l[val2_as[p, j]]] = val2_index[j, val2_l[val2_as[p, j]]] + 1 / (val2_s[p, j] + 0.001)
                    val3_index[j, val3_l[val3_as[p, j]]] = val3_index[j, val3_l[val3_as[p, j]]] + 1 / (val3_s[p, j] + 0.001)
                    val4_index[j, val4_l[val4_as[p, j]]] = val4_index[j, val4_l[val4_as[p, j]]] + 1 / (val4_s[p, j] + 0.001)
                    val5_index[j, val5_l[val5_as[p, j]]] = val5_index[j, val5_l[val5_as[p, j]]] + 1 / (val5_s[p, j] + 0.001)
            acc[k_index == k * np.ones(k_index.shape[0])] = np.mean(
                [np.mean(f1_y.T == np.argmax(val1_index, axis=1)), np.mean(f2_y.T == np.argmax(val2_index, axis=1)),
                 np.mean(f3_y.T == np.argmax(val3_index, axis=1)), np.mean(f4_y.T == np.argmax(val4_index, axis=1)),
                 np.mean(f5_y.T == np.argmax(val5_index, axis=1))])
        print('5fold cross validation was successfully completed (L2 norm)')

        return acc

    def predict(self, X, k):
#add predict function
        num_train = self.Xtr.shape[0]
#number of train data
        num_test = X.shape[0]
#number of test data
        distances_L1 = np.zeros((num_train, num_test))
        distances_L2 = np.zeros((num_train, num_test))
        for i in range(num_test):
            
#calculate distances
            distances_L1[:, i] = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
#L1 metric
            distances_L2[:, i] = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis = 1))
#L2 metric
        pred_index_L1 = np.zeros((num_test, 10))
        pred_index_L2 = np.zeros((num_test, 10))
        distances_L1_s = np.sort(distances_L1, axis=0)        
#sorting L1 distances
        distances_L2_s = np.sort(distances_L2, axis=0)
#sorting L2 distances
        distances_L1_as = np.argsort(distances_L1, axis=0)
#argument sorting of L1 distances
        distances_L2_as = np.argsort(distances_L2, axis=0)
#argument sorting of L2 distances
        for p in range(k):
#take k'th data
            for j in range(num_test):
#input weight information which is number of one divided by distance in pred_index
                pred_index_L1[j, self.ytr[distances_L1_as[p, j]]] = pred_index_L1[j, self.ytr[distances_L1_as[p, j]]] + 1 / (distances_L1_s[p, j] + 0.001)
                pred_index_L2[j, self.ytr[distances_L2_as[p, j]]] = pred_index_L2[j, self.ytr[distances_L2_as[p, j]]] + 1 / (distances_L2_s[p, j] + 0.001)
        Ypred_L1 = np.argmax(pred_index_L1, axis = 1).T
#finally make list of labels(L1)
        Ypred_L2 = np.argmax(pred_index_L2, axis = 1).T
#list of labels(L2)

        return Ypred_L1, Ypred_L2
#take pridict labels data

nn = NearestNeighbor()
#make class
k = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
#set k number
acc_L1 = nn.cval_5f_L1(Xtr1, Xtr2, Xtr3, Xtr4, Xtr5, Ytr1, Ytr2, Ytr3, Ytr4, Ytr5, k)
#5fold cross validation for L1 metric
acc_L2 = nn.cval_5f_L2(Xtr1, Xtr2, Xtr3, Xtr4, Xtr5, Ytr1, Ytr2, Ytr3, Ytr4, Ytr5, k)
#5fold cross validation for L2 metric
print(acc_L1)
print(acc_L2)
if k.shape[0] > np.argmax(np.append(acc_L1, acc_L2, axis = 0)):
#code for choose K value and metric
    L_best = 1
    k_best = k[np.argmax(acc_L1)]
else:
    L_best = 2
    k_best = k[np.argmax(acc_L2)]

print('Best distance method : L%d' % (L_best))
print('Best K value : %d' % (k_best))

nn.train(Xtr, Ytr)
#train data
[Ytr_predict_L1, Ytr_predict_L2] = nn.predict(Xte, k_best)
#predict test data
print('accuracy_L1 : %f' % (np.mean(Ytr_predict_L1 == Yte)))
print('accuracy_L2 : %f' % (np.mean(Ytr_predict_L2 == Yte)))

