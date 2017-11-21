import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class rbm():
    def __init__(self,shape):
        self.shape=shape#the shape of rbm_w

    def sample_prob(self,probs):
        return tf.nn.relu(tf.sign(probs-tf.random_uniform(tf.shape(probs))))#according to the probability to decide whether to activate the unit

    def propup(self,v,w,a):
        return tf.nn.sigmoid(tf.add(tf.matmul(v, w), a))#calculate the hidden layer

    def propdown(self,h,w,b):
        return tf.nn.sigmoid(tf.add(tf.matmul(h,tf.transpose(w)),b))#calculate the reconsitution of v


    def train(self,data,label,train_step,layer_num,learn_rate=1):
        x_in=tf.placeholder(tf.float32,[None,self.shape[0]])

        rbm_w = tf.placeholder(tf.float32, self.shape)
        rbm_a = tf.placeholder(tf.float32, self.shape[-1])
        rbm_b = tf.placeholder(tf.float32, self.shape[0])

        '''
        use CD algorithm to update w,a,b
        '''
        h_start = self.sample_prob(self.propup(x_in,rbm_w,rbm_a))
        v_end = self.sample_prob(self.propdown(h_start,rbm_w,rbm_b))
        h_end = self.propup(v_end,rbm_w,rbm_a)
        up_w = tf.matmul(tf.transpose(x_in), h_start)
        down_w = tf.matmul(tf.transpose(v_end), h_end)
        w = rbm_w + learn_rate * (up_w - down_w) / tf.to_float(tf.shape(x_in)[0])
        a = rbm_a + learn_rate * tf.reduce_mean((h_start - h_end), 0)
        b = rbm_b + learn_rate * tf.reduce_mean((x_in - v_end), 0)

        h_pros = self.propup(x_in,w,a)
        h = self.sample_prob(h_pros)#hidden layer state
        x_pros = self.propdown(h,w,b)
        x_decoder = self.sample_prob(x_pros)#reconsitution v

        error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x_in - x_decoder), 1)))#calculate error
        with tf.Session() as sess:
            n_w = np.zeros(self.shape, np.float32)
            n_b = np.zeros(self.shape[0], np.float32)
            n_a = np.zeros(self.shape[-1], np.float32)
            sess.run(tf.global_variables_initializer())
            errors=[]
            for i in range(train_step):
                x=data[i*100:i*100+100][:]
                feed_dict={x_in:x,rbm_w:n_w,rbm_a:n_a,rbm_b:n_b}
                n_w=sess.run(w,feed_dict=feed_dict)
                n_a=sess.run(a,feed_dict=feed_dict)
                n_b=sess.run(b,feed_dict=feed_dict)

                if i % 10 == 0:
                    Error=sess.run(error,feed_dict=feed_dict)
                    errors.append(Error)
                    print('number %d error is %f' % (i, Error))
            plt.plot(errors)
            plt.xlabel('train_step')
            plt.ylabel('error')
            plt.title('rbm_layer'+str(layer_num+1))
            plt.show()
            x,y=data,label
            h_result=sess.run(h,feed_dict={x_in:x,rbm_w:n_w,rbm_a:n_a,rbm_b:n_b})
            print h_result.shape,label.shape
            return h_result,label



