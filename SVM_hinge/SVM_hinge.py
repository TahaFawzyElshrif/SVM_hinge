import numpy as np
from sklearn.utils import check_array
import random


def calc_loss_hingePoint(x,y,w,b):
    return max(0,1-y*(np.dot(x,w)+b))


def calc_regulize(w):
    return sum(abs(i) for i in w)


def loss_hinge_wb(x,y_true,w,b,lambada):#take whole data ,x must be 2 d numpy array and y is list
    y_true_list = check_array(np.array(y_true).reshape(-1,1))#reshaped to be 2d like [[-1],[1]]  to could be used in check_array ,check_array is to Convert y_true (assuming it's a NumPy array) to a list for SymPy operations
    total_loss = 0
    for x_i,y_i in zip(x,y_true_list):
        total_loss+=calc_loss_hingePoint(x_i,y_i,w,b)
    total_loss=float(total_loss) /( 2*len(y_true))
    total_loss=total_loss+((lambada*calc_regulize(w))/( 2*len(y_true))) #regulization part
    return  total_loss


def sub_gradient(x_i,y_i,w,b):#return new w,b  ,x_i ,w is vector one row ,b,y_i const 
    current_val=1-(y_i*(np.dot(x_i,w)+b))
    if (current_val>0):
        df_dw=[-y_i*i for i in x_i] #divide by 2m is done in total subgradient
        df_db=-1
        return df_dw+[df_db]
    elif (current_val<0):
        return [0]*((len(w)+1))
    else : # equal zero
        df_dw_larger_zero=[-y_i*i for i in x_i]+[-1]#same case as when current_val>0
        df_dw_equal_zero=[random.uniform(range(0,max(i,0))) for i in df_dw_larger_zero] #i may be > j like df_db in first case=-1 ,in second case=0
        #uniform to choose random float
        return df_dw_equal_zero
    

def regulize_wi_grad(w_i,lambada,m):
    loss_i=0
    if (w_i>0):
        loss_i=1
    elif (w_i<0):
        loss_i=-1
    else:#w_i=0 ,subgradient case 
        loss_i=random.uniform(-1,1)
    loss_i=(loss_i*lambada)/(2*m)
    return loss_i

def sub_gradient_all_data(x,y,w,b,lambada):#return array
    m=len(y)
    df_dw = np.array([0 for i in range(len(w))])
    df_db = 0
    for x_i,y_i in zip(x,y):
        grad=sub_gradient(x_i,y_i,w,b)
        df_dw=(np.add(grad[:-1],df_dw))
        df_dw=np.add(df_dw,[regulize_wi_grad(w_i,lambada,m) for w_i in w])
        df_db=(df_db+int(grad[-1]))
    df_dw=df_dw/(2*m)
    df_db=df_db/(2*m)
    
    return (df_dw,df_db)


def sub_gradient_descent(x,y,lring_rate=.0001, num_iters=200,lambada=1):
    w=[0 for i in range(x.shape[1])] #intial zeros
    b=0


    last_cost_history=[100000000]*10
    index_last_cost_history=0
    
    
    for i in range(num_iters):
        sub_gradient=sub_gradient_all_data(x,y,w,b,lambada)
        df_dw=sub_gradient[0]
        df_db=sub_gradient[1]
        w=np.subtract(w,df_dw*lring_rate)
        b=b-df_db*lring_rate    
    
    
        if (loss_hinge_wb(x,y,w,b,lambada)>np.mean(last_cost_history)):#to ensure not getting bad after converaging ,compare with average of last 10 iterations
            #order not important here of the circluar list ,just want average
            break
        else:
            last_cost_history[index_last_cost_history]=loss_hinge_wb(x,y,w,b,lambada)
            index_last_cost_history=(index_last_cost_history+1)%10

        if (i % 20 ==0):
            print (f"iteration {i} finished , cost {loss_hinge_wb(x,y,w,b,lambada)}  , w={w}   , b= {b}")
    print (f"last iteration finished , cost {loss_hinge_wb(x,y,w,b,lambada)}  , w={w}   , b= {b}")

    return (w,b)


class SVM_hinge:

    def __init__(self):  
        """
        init model
        """
        pass

    
    def fit(self, x,y,lring_rate=.0001,num_iters=200,lambada=1):
        """
        The main method to fit model ,it train the model on data and set the parameters
        x : features
        y : target variable ,must classes -1,1 (for this version)
        lring_rate : learining rate for the optimizer used ,this version use gredient descent ,default .0001
        num_iters : max number of iterations , default : 200
        lambada :regulaization parameter for first norm optimization ,default 1 (no regualize)
        
        """
        

        if (set(y)!={1,-1}):
            raise("Y must be only 1 , -1")


        self.x = x
        self.y = y
        self.lring_rate = num_iters
        sub_gradient=sub_gradient_descent(x,y,lring_rate, num_iters,lambada)
        self.w=sub_gradient[0]
        self.b=sub_gradient[1]

    def predict(self,x):
        """
        Predict class  ,this method work by just multiply fitted weights with x then predict
        """
        points_value=((np.dot(x,self.w))+self.b)#must be (x,w) not (w,x)
        return [1 if i>=1 else -1 for i in points_value]

