import numpy as np


#x_data = np.array([1,2,3,4,5]).reshape(5,1)
#t_data = np.array([2,3,4,5,6]).reshape(5,1)

loaded_data = np.loadtxt('./data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]


W = np.random.rand(3,1) #3X1 행렬
b = np.random.rand(1)

print("W = ", W, ", b = ",b)

def loss_func(x, t):
    y = np.dot(x, W) + b
    return (np.sum( (t-y)**2 )) / ( len(x) )

def predict(x):
    y = np.dot(x,W) + b

    return y

#f는 다변수 함수, x는 모든 변수를 포함하고 있는 numpy 객체(벡터, 행렬)
def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x) # 계산된 수치미분 값 저장 변수

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        # numpy 타입은 mutable이기 때문에 원래 값 보관
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x) # f(x-delta_x)

        #하나의 변수에 대한 수치미분 계산 후, 결과값 저장
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

learning_rate = 1e-5 #1e-2, 1e-3 은 손실함수 값 발산

f = lambda x : loss_func(x_data, t_data)

for step in range(20001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if(step % 400 == 0):
        print("step = ", step, "error value = ", loss_func(x_data,t_data), " W = ",W, ", b = ",b)

test_data = np.array([100, 98, 81])
print("predicted value : ", predict(test_data))