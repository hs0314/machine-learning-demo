import tensorflow as tf
import numpy as np

loaded_data = np.loadtxt('./data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[:, 0:-1]
y_data = loaded_data[:, [-1]]

'''
tmp = x_data[:, 0:1]
print(tmp.reshape(-1,).tolist())
#print(tmp.shape)
'''

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val,  _ = sess.run([cost, hypothesis, train],
                                         feed_dict={x: x_data, y: y_data})

    if step % 10 == 0:
        print(step, "Costs: ", cost_val, "\nPrediction:\n", hy_val)

print("predicted score : ", sess.run(hypothesis, feed_dict={x: [[100,70,101]]}))