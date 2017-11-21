import tensorflow as tf
import matplotlib.pyplot as plt
'''
linear_layer
be used to classify
'''
class dbn_linear():
    def __init__(self,shape):
        self.shape=shape#[input units,output_units]
    def train(self,data,label,train_step):
        x=tf.placeholder(tf.float32,[None,self.shape[0]])#the data after encoder
        y=tf.placeholder(tf.float32,[None,self.shape[1]])#label

        y_=tf.contrib.layers.fully_connected(x,10,activation_fn=None)
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
        optimizer=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print label.shape
            cross_entropys=[]
            for i in xrange(train_step):
                batch_xs=data[i*100:i*100+100][:]
                batch_ys=label[i*100:i*100+100][:]
                sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})
                if i%10==0:
                    Cross_entropy=sess.run(cross_entropy,feed_dict={x:batch_xs,y:batch_ys})
                    cross_entropys.append(Cross_entropy)
                    print('number %d cross_entropy is %f'%(i,Cross_entropy))
                    print("number %d accuracy is %f" % (i, sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})))
            '''
            plot the change of cross_entropy
            '''

            plt.plot(cross_entropys)
            plt.xlabel('train_step')
            plt.ylabel('cross_entropy')
            plt.title('linear_layer')
            plt.show()