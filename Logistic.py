import numpy as np
import matplotlib.pyplot as plt
import csv

f1 = open('./data/train_input.csv', 'r')
f2 = open('./data/train_output.csv', 'r')
f3 = open('./data/test_input.csv', 'r')
f4 = open('./data/test_output.csv', 'r')

r1 = csv.reader(f1)
r2 = csv.reader(f2)
r3 = csv.reader(f3)
r4 = csv.reader(f4)

train_input = []
train_output = []
test_input = []
test_output = []

for row in r1:
    train_input.append(row)
for row in r2:
    train_output.append(row)

for row in r3:
    test_input.append(row)
for row in r4:
    test_output.append(row)


f1.close()
f2.close()
f3.close()
f4.close()

'''
first_image = train_input[1]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
'''

def do_sigmoid(beta, x):
    betat = beta.transpose()
    mult = np.matmul(beta, x)
    return 1 / (1 + np.exp(-mult))


l = len(train_input)
X = train_input
X_test = test_input



for i in range(l):
    X[i] = list(map(float,X[i]))
    train_output[i] = list(map(float,train_output[i]))

for i in range(len(X_test)):
    X_test[i] = list(map(float, X_test[i]))
    test_output[i] = list(map(float,test_output[i]))

y = []
for i in range(l):
    if train_output[i][0] == 1 :
        y.append(1)
    else :
        y.append(0)

y_test = []
for i in range(len(X_test)):
    if test_output[i][0] == 1 :
        y_test.append(1)
    else :
        y_test.append(0)

y = np.array(y)
X = np.array(X)

#print(X.shape)
#print(y.shape)
#print(type(train_output[0][0]))



dim = np.prod(X[0].shape)
data_num = int( np.prod(X.shape) / dim )

#print("dim : ", dim)
#print("data : ", data_num)
temp_beta = np.zeros(dim)

learn_rate = 0.001

#print(type(X[0][0]))

for i in range(100):
    print("iter ",i+1)
    small_step = 0
    for j in range(data_num):
        val1 = do_sigmoid(temp_beta, X[j])
        val2 = (y[j]-val1) * X[j]
        small_step = small_step + val2
    temp_beta = temp_beta + learn_rate * small_step

#print(temp_beta)


y_predlist = []
for i in range(len(X_test)):
    y_pred = do_sigmoid(temp_beta, X_test[i])
    if(y_pred < 0.5):
        y_predlist.append(0)
    else:
        y_predlist.append(1)

cnt = 0
for i in range(len(y_predlist)):
    if y_predlist[i] == y_test[i] :
        cnt = cnt + 1
    else : pass
print(float(cnt/len(y_predlist)))



'''
for k in range(2) :
    for j in range(dim):
        small_step = 0
        val1 = do_sigmoid(temp_beta, X[j])
        for i in range(data_num):
            val1 = do_sigmoid(temp_beta, X[j])
            val2 = (y[j]-val1) * X[j]
            small_step = small_step + val2
        temp_beta = temp_beta + learn_rate * small_step

print(temp_beta)
'''
