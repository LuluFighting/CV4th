import random
import cv2
INTIAL_INLIER_NUM = 4
MEAN_ERROR = 1E-2
def ransacMatching(A, B,k):  #k为轮询次数
    #A & B: List of List
    if len(A)!=len(B):
        print('the length of two list is not same')
        exit(-1)
    numOfPoint = len(A)
    inlierA,inlierB,outlier=[],[],None
    for _ in range(INTIAL_INLIER_NUM): #随机取内点
        rand_i = random.randint(0,numOfPoint-1)
        rand_j = random.randint(0,numOfPoint-1)
        inlierA.append(A[rand_i])
        inlierB.append(B[rand_j])
        A.pop(rand_i)
        B.pop(rand_j)
    for epoch in range(k):
        homograph = cv2.findHomography(inlierA,inlierB)
        flag = False   #标记次轮是否在外点集合中找到了匹配的点
        removeListOfA,removeListOfB = [],[]
        for point in A:         #找到homograph矩阵后，进行测试
            src_point = homograph*point
            for fact_point in B:
                if abs(fact_point-src_point)<=MEAN_ERROR: #如果比误差要小,可以认为拟合了点
                    flag=True
                    inlierB.append(fact_point)
                    inlierA.append(point)
                    removeListOfA.append(point)
                    removeListOfB.append(fact_point)
        for element in removeListOfA:
            A.remove(element)
        for element in removeListOfB:
            B.remove(element)
        if flag is not True:  #点没有改变
            break
    return inlierA,inlierB