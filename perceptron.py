import matplotlib.pyplot as plt
import numpy as np
import csv

iterations = 200
x = []
gram = np.zeros(shape=(150, 150))
alpha = np.zeros(shape=(150, 1))
b = 0
delta = 0.5

with open('flower_data.csv', 'r') as f:
    lines = csv.reader(f)
    for line in lines:
        try:
            x.append([float(line[0]), float(line[1])])
        except ValueError:
            pass

x = np.array(x)
y = [-1 for i in range(50)] + [1 for i in range(100)]
for index_i, xi in enumerate(x):
    xi = np.array(xi)
    for index_j, xj in enumerate(x):
        xj = np.array(xj)
        gram[index_i][index_j] = xi.dot(xj.T)

# print(gram)
# print(y)

for c in range(iterations):
    for index_j, placeholder in enumerate(y):
        fx = np.dot(alpha.T * y, gram[:, index_j]) + b
        if y[index_j]*fx <= 0:
            alpha[index_j] += delta
            b += delta*y[index_j]

# print(alpha)

w = np.zeros(shape=(1, 2))

for index, ele in enumerate(alpha):
    w = w + ele*x[index]*y[index]
    print(w)


k = -w[0][0] / w[0][1]
c = -b / w[0][1]
print(k, c)

classifier_x = np.linspace(start=3.5, stop=8, num=10)
classifier_y = k*classifier_x + c

plt.figure(figsize=(10, 6.18))
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(x[50:, 0], x[50:, 1], color='blue', marker='+', label='Iris-virginica')
plt.xticks(np.linspace(start=3, stop=8, num=10))
plt.yticks(np.linspace(start=2, stop=5, num=10))
plt.xlim(x[:, 0].min(), x[:, 0].max())
plt.ylim(2, 5)
plt.xlabel('petal')
plt.ylabel('scape')
plt.plot(classifier_x, classifier_y, 'green', '--')
plt.legend(loc='upper fit')
plt.annotate('f(x) = sign[{k:.6f} * x + ({c:.6f})]'.format(k=k, c=c), xy=(5.5, 4.8))
# plt.savefig('./flower_classify.jpg')
plt.show()






