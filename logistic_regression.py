import numpy as np

x_data = np.array([[2,4],[4,11],[6,6],[8,5],[10,7],[12,16],[14,8],[16,3],[18,7]])
y_data = np.array([0,0,0,0,1,1,1,1,1]).reshape(9,1)

# 임의의 w, b값 설정
W = np.random.rand(2,1)
b = np.random.rand(1)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def loss_func(x, t):
    delta = 1e-7 # log 무한대 발산 방지

    z = np.dot(x,W) + b #linear regression 결과 값
    y = sigmoid(z)

    #cross-entropy
    return -np.sum( t*np.log(y + delta) + (1-t)*np.log((1-y)+delta))

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

learning_rate = 1e-2
f = lambda x : loss_func(x_data, y_data)

for step in range(80001):

    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if step % 800 ==0:
        print("step : ",step, "loss_func_val : ", loss_func(x_data, y_data), "W : ",W, "b : ",b)

# 테스트 케이스를 통해서 임의의 x1, x2에 대한 P/F을 알 수 있다.
tc1 = np.array([[3,3], [10,10], [100,100], [5,5]])
tc1_z = np.dot(tc1, W) + b
tc1_y = sigmoid(tc1_z)

print(tc1_y)