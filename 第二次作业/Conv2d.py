import numpy as np
import cv2

def medianBlur(img, kernel, padding_way):
    #img & kernel is List of List; padding_way a string
    def getMedianArray(img):
        height,width = img.shape[0],img.shape[1]
        #we assume that stride is 1,then:
        padding_height = (kernel.shape[0]-1)//2           #compute the padding height
        padding_width = (kernel.shape[1]-1)//2            #compute the padding width
       #print(img)
        padding_img = None
        if padding_way == 'REPLICA':
            col_left,col_right = img[:,:1],img[:,width-1:]
            col_left,col_right = np.squeeze(col_left),np.squeeze(col_right)
            for i in range(padding_width):
                padding_img = np.insert(img,0,col_left,axis=1)
                padding_img = np.insert(padding_img,width+2*i+1,col_right,axis=1)
            row_up,row_down = padding_img[:1,:],padding_img[height-1:,:]
            row_up,row_down = np.squeeze(row_up),np.squeeze(row_down)
            for i in range(padding_height):
                padding_img = np.insert(padding_img,0,row_up,axis=0)
                padding_img = np.insert(padding_img,height+2*i+1,row_down,axis=0)
            # print(padding_img)
        if padding_way == 'ZERO':
            col = np.zeros(img.shape[0])
            for i in range(padding_width):
                padding_img = np.insert(img,0,col,axis=1)
                padding_img = np.insert(padding_img,width+2*i+1,col,axis=1)
            row = np.zeros(padding_img.shape[1])
            for i in range(padding_height):
                padding_img = np.insert(padding_img,0,row,axis=0)
                padding_img = np.insert(padding_img,height+2*i+1,row,axis=0)
            # print(padding_img)
        result = [[0 for _ in range(width)] for _ in range(height)]
        # print(padding_img.shape)
        for i in range(height):
            for j in range(width):
                # if i+kernel.shape[0]>img.shape[0] or j+kernel.shape[1]>img.shape[1]:
                #     continue
                temp = padding_img[i:i+kernel.shape[0],j:j+kernel.shape[1]]
                result[i][j] = np.median(temp)
        result =  np.array(result)
        print(result.shape)
        return result
    height,width,channel = img.shape
    if kernel.shape[0]%2==0:
        print('kernel size must be odd')
        exit(-1)
    if padding_way not in ['ZERO','REPLICA']:
        print('padding way must be ZERO or REPLICA')
        exit(-1)
    if channel==1:
        #img2D = np.reshape(img,(height,width))
        img_median = medianBlur(img,kernel,padding_way)
        return img_median
    elif channel!=3:
        print('the channel of img must be 1 or 3')
        exit(-1)
    else:
        B,G,R = cv2.split(img)
        print(B.shape)
        print(G.shape)
        B_res = getMedianArray(B)
        G_res = getMedianArray(G)
        R_res = getMedianArray(R)
        print('I am here')
        img_median = cv2.merge((B_res,G_res,R_res))
        return img_median
# img = np.array([
#     [1,2,3,4],
#     [5,6,7,8],
#     [9,10,11,12],
# ])
kernel = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
img = cv2.imread('../lenna.jpeg')
cv2.imshow('origin',img)
img_median = medianBlur(img,kernel,'REPLICA')
cv2.imshow('hello',img_median)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
