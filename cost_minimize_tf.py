import tensorflow as tf
import matplotlib.pyplot as plt

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = X * W # y = Wx
cost = tf.reduce_mean(tf.square(hypothesis - Y))

####Minimize : W -= learning_rate * derivative

'''최종적으로 다음과 같다
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
tf.Session().run(train)
'''

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
#####

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

x_data = [1,2,3]
y_data = [1,2,3]

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))