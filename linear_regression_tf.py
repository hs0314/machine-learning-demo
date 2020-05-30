import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = x * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={x: [1, 2, 3,4,5], y: [2.1, 3.1, 4.1,5.1,6.1]})

    #if step % 20 == 0:
        #print(step, sess.run(cost), sess.run(W), sess.run(b))
        #print(step, cost_val, W_val, b_val)


print(sess.run(hypothesis, feed_dict={x: [5]} ))
print(sess.run(hypothesis, feed_dict={x:[1.5, 3.5]}))
