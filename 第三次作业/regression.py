import numpy as np
from matplotlib import pyplot as plt
import random

#定义一个线性回归类
class Regression:
    #随机初始化权重值
    def __init__(self,input_dim,output_dim=1):
        self.weight = np.random.rand(input_dim,output_dim)
        self.bias = np.random.rand()
    #每次读入batch_size的数据
    def dataLoader(self,x,y,batch_size):
        idx = np.random.choice(len(x),batch_size)
        batch_x = [x[i] for i in idx]
        batch_y = [y[i] for i in idx]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x.reshape(len(batch_x),1),batch_y.reshape(len(batch_y),1)
    #进行正向传播
    def forward(self,x):
        return x@self.weight+self.bias
    #计算loss函数
    def computeCost(self,x,y):
        y_pred = self.forward(x)
        diff = (y_pred - y).T*(y_pred - y)
        return (diff.sum()/len(x))
    #计算梯度
    def gradient(self,x,y):
        y_pred = self.forward(x)
        dw = ( x.T @ (y_pred - y)) / len(x)
        db = np.sum(y_pred-y)/len(x)
        return dw,db
    #梯度下降，默认学习率为0.00001，EPOCH为1000
    def gradientDescent(self,x,y,BATCH_SIZE=64,LR=0.00001,EPOCH=1000,):
        loss_list = []
        for epoch in range(EPOCH):
            batch_x,batch_y = self.dataLoader(x,y,BATCH_SIZE)
            dw,db = self.gradient(batch_x,batch_y)
            self.weight -= LR*dw
            self.bias -= LR*db
            loss = self.computeCost(batch_x,batch_y)
            loss_list.append(loss)
            if epoch%50 == 0 :
                print('EPOCH: {:03d}_|| Loss: {:.5f}|| '.format(epoch,loss))
        #画出loss和iteration的图像
        plt.plot(np.arange(EPOCH),loss_list,'r')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Training Epoch')
        plt.show()
    #生成数据
    def generateData(self,output_dim,input_dim):
        w = np.random.rand(output_dim,input_dim) + random.random()  # for noise random.random[0, 1)
        b = np.random.rand() + random.random()
        x = np.arange(1,101) + np.random.rand()
        x = x.reshape((x.shape[0], 1))
        y = w*x+ b + random.random() * random.randint(-1, 1)
        plt.scatter(x,y)
        plt.show()
        return x,y
def run():
    regress = Regression(1)
    x,y = regress.generateData(1,1)
    regress.gradientDescent(x,y)
if __name__ == '__main__':
    run()