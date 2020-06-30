import paddle.fluid as fluid
import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = act_dim*128
        hid2_size = act_dim*128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

        # self.conv1 = layers.conv2d(num_filters=32,filter_size=3,padding="same",act='relu')
        # self.pool1 = layers.pool2d(pool_size=2, pool_type="max", pool_stride=2, )
        # self.conv2 = layers.conv2d(num_filters=16, filter_size=3, padding="same", act='relu')
        # self.pool2 = layers.pool2d(pool_size=2, pool_type="max", pool_stride=2, )
        # self.conv3 = layers.conv2d(num_filters=8, filter_size=3, padding="same", act='relu')
        # self.pool3 = layers.pool2d(pool_size=2, pool_type="max", pool_stride=2, )
        # self.fc4 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)

        # x=self.conv1(img)
        # x=self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.fc4(x)
        # Q=Q +x

        return Q
