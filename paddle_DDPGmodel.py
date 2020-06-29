import paddle.fluid as fluid
import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid_size1 = 20
        hid_size2 = 20
        hid_size3 = 20

        #self.conv2 = layers.conv2d(num_filters=10, filter_size=(10, 6), stride=6, act='relu')

        self.fc1 = layers.fc(size=hid_size1, act='relu')
        self.fc2 = layers.fc(size=hid_size2, act='relu')
        self.fc3 = layers.fc(size=hid_size3, act='relu')
        self.fc4 = layers.fc(size=act_dim, act='softmax')

    def policy(self, obs):
        #obs = self.conv2(obs)
        hid = self.fc1(obs)
        means = self.fc2(hid)
        means = self.fc3(means)
        means = self.fc4(means)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid_size1 = 20
        hid_size2 = 20
        hid_size3 = 20

        #self.conv2 = layers.conv2d(num_filters=10, filter_size=(10, 6), stride=6, act='relu')
        self.fc1 = layers.fc(size=hid_size1, act='relu')
        self.fc2 = layers.fc(size=hid_size2, act='relu')
        self.fc3 = layers.fc(size=hid_size3, act='relu')
        self.fc4 = layers.fc(size=1, act='softmax')

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        #concat = self.conv2(concat)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = self.fc3(Q)
        Q = self.fc4(Q)

        Q = layers.squeeze(Q, axes=[1])
        return Q