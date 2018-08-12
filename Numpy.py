import numpy as np
print(np.__version__)
#一维数组
np.array([1,2,3])
#二维数组
np.array([(1,2,3),(4,5,6)])
print(np.zeros((3,3)))
np.ones((2,3,4))
np.arange(5)
#创建二维数组
np.arange(6).reshape(2,3)
np.eye(3)
np.linspace(1, 10, num=6)
np.random.rand(2,3)
#创建二维随机数组（小于5）
np.random.randint(5, size=(2,3))
np.fromfunction(lambda i, j: i + j, (3, 3))
a = np.array([10,20,30,40,50])
b = np.arange(1,6)
a + b
a * b
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
#矩阵元素乘
A * B
np.dot(A, B)
# 如果使用 np.mat 将二维数组准确定义为矩阵，就可以直接使用 * 完成矩阵乘法计算
np.mat(A) * np.mat(B)
print(A.T)
#矩阵求逆
np.linalg.inv(A)
print(a)
np.sin(a)
np.exp(a)
np.sqrt(a)
np.power(a, 3)
a = np.array([1, 2, 3, 4, 5])
a[0], a[-1]
a[0:2], a[:-1]
#二维数组索引
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
a[0], a[-1]
a[:,1]
a[1:3, :]
a = np.random.random((3, 2))
a.shape
a.reshape(2, 3)
a.resize(2, 3)
#展平数组
print(a.ravel())
a = np.random.randint(10, size=(3, 3))
b = np.random.randint(10, size=(3, 3))
#垂直合并数组
np.vstack((a, b))
np.hstack((a, b))
np.hsplit(a, 3)
np.vsplit(a, 3)
a = np.array(([1,4,3],[6,2,9],[4,7,2]))
#返回每列最大值
np.max(a, axis=0)
np.min(a, axis=1)
#返回每列最大值索引
np.argmax(a, axis=0)
np.argmin(a, axis=1)
np.median(a, axis=0)
np.mean(a, axis=1)
np.average(a, axis=0)
np.var(a, axis=1)
np.std(a, axis=0)