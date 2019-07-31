import numpy as np
import random
from matplotlib import pyplot as plt
class Classfication:
    #初始化权重
    def __init__(self,input_dim,output_dim=1):
        self.weight = np.random.rand(input_dim,output_dim)
        self.bias = np.random.rand()
    #读入数据
    def dataLoader(self,x,y,batch_size):
        idx = np.random.choice(len(x),batch_size)
        batch_x = [x[i] for i in idx]
        batch_y = [y[i] for i in idx]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x.reshape(len(batch_x),2),batch_y.reshape(len(batch_y),1)
    #激活函数
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    #前向计算
    def forward(self,x):
        z = x@self.weight + self.bias
        return self.sigmoid(z)
    #计算loss
    def computeCost(self,x,y):
        a = self.forward(x)
        return -(np.mean(np.multiply(y,np.log(a))+np.multiply(1-y,np.log(1-a))))
    #计算梯度
    def gradient(self,x,y):
        a = self.forward(x)
        dw = x.T@(a-y)
        db = a-y
        return dw,db
    #梯度下降
    def gradientDescent(self,x,y,LR=0.003,BATCH_SIZE=32,EPOCH=1000):
        loss_list = []
        for epoch in range(EPOCH):
            batch_x, batch_y = self.dataLoader(x, y, BATCH_SIZE)
            dw, db = self.gradient(batch_x, batch_y)
            self.weight -= LR * dw
            self.bias -= LR * db
            loss = self.computeCost(batch_x, batch_y)
            loss_list.append(loss)
            if epoch % 50 == 0:
                print('EPOCH: {:03d}_|| Loss: {:.5f}|| '.format(epoch, loss))
        # 画出loss和iteration的图像
        plt.plot(np.arange(EPOCH), loss_list, 'r')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Training Epoch')
        plt.show()
    #产生一个二分类的数据
    def generateData(self):
        n_data = np.ones((100,2))
        #从均值为2*n_data，std=1的标准正态分布中随机生成元素
        x0 = np.random.normal(2*n_data,1)
        y0 = np.zeros(100)
        x1 = np.random.normal(-2*n_data,1)
        y1 = np.ones(100)
        x = np.concatenate((x0,x1),axis=0)
        y = np.concatenate((y0,y1),axis=0)
        plt.scatter(x[:,0],x[:,1],c=y,s=100)
        plt.show()
        return x,y
def run():
    classification = Classfication(2,1)
    x,y = classification.generateData()
    classification.gradientDescent(x,y)
if __name__ == '__main__':
    run()