import torch
import torchvision as ptv
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
# import numpy as np

train_set = ptv.datasets.MNIST("./data",
                               train=True,
                               transform=ptv.transforms.ToTensor(),
                               download=True)
test_set = ptv.datasets.MNIST("./data",
                              train=False,
                              transform=ptv.transforms.ToTensor(),
                              download=True)

# MINIST数据集中图像数据存放在data中，其对应的答案放在targets属性中。
# print(train_set.data.size())
# print(train_set.targets.size())

# 每100个数据为一组，进行一次迭代
learning_rate = 0.01
EPOCH = 5
Batch_size = 64
PATH_to_log_dir = "./runs"
train_dataset = torch.utils.data.DataLoader(train_set, batch_size=Batch_size)
test_dataset = torch.utils.data.DataLoader(test_set, batch_size=Batch_size)
device = torch.device('cuda:0')
writer = SummaryWriter(PATH_to_log_dir)


class MLP():
    input_size = 784
    hidden1_size = 390
    hidden2_size = 180
    hidden3_size = 90
    output_size = 10
    W = []
    b = []
    z = []
    a = []
    d_W = []
    d_b = []

    # 必须给定一个self参数
    def sig(self, x):
        return 1 / (torch.exp(-x) + 1)

    def ReLU(self, x):
        y = (torch.clamp(x, min=0)).float()
        return y

    def d_ReLU(self, x):
        y = (x > 0).float()
        return y

    def init_parameters(self):
        self.W.append(
            torch.randn(self.hidden1_size, self.input_size) /
            math.sqrt(28 * 28))
        self.W.append(
            torch.randn(self.hidden2_size, self.hidden1_size) /
            math.sqrt(self.hidden1_size * self.hidden1_size))
        self.W.append(
            torch.randn(self.hidden3_size, self.hidden2_size) /
            math.sqrt(self.hidden2_size * self.hidden2_size))
        self.W.append(
            torch.randn(self.output_size, self.hidden3_size) /
            math.sqrt(self.hidden3_size * self.hidden3_size))
        self.b.append(torch.randn(self.hidden1_size, 1) / math.sqrt(28 * 28))
        self.b.append(torch.randn(self.hidden2_size, 1) / math.sqrt(self.hidden1_size * self.hidden1_size))
        self.b.append(torch.randn(self.hidden3_size, 1) / math.sqrt(self.hidden2_size * self.hidden2_size))
        self.b.append(torch.randn(self.output_size, 1) / math.sqrt(self.hidden3_size * self.hidden3_size))

    def forward_pro(self, x):
        self.z.clear()
        self.a.clear()

        self.z.append((self.W[0].mm(x) + self.b[0]))
        self.a.append(torch.relu(self.z[0]))
        # self.a.append(self.sig(self.z[0]))

        self.z.append(self.W[1].mm(self.a[0]) + self.b[1])
        self.a.append(torch.relu(self.z[1]))
        # self.a.append(self.sig(self.z[1]))

        self.z.append(self.W[2].mm(self.a[1]) + self.b[2])
        self.a.append(torch.relu(self.z[2]))
        # self.a.append(self.sig(self.z[2]))

        self.z.append(self.W[3].mm(self.a[2]) + self.b[3])
        self.a.append(torch.nn.functional.softmax(
            self.z[3], dim=0))  # exp(x)之和不是所有的exp(x)之和，是单个样本的exp(x)之和

        # print(self.z[2])
        # print(self.a[2])
        # print(self.z[3])
        # print(self.a[3])
        # exit()
        return self.a[3]

    def loss_cal(self, y_hat, _class):
        # print(y_hat)
        return -1 * torch.log(y_hat[_class])

    def backward_pro(self, x, t):
        self.d_W.clear()
        self.d_b.clear()

        # delta 是 loss 对 z 的导数
        # 输出层，delta求比较特殊
        delta = self.a[3] - t
        self.d_W.append(delta.mm(self.a[2].T))
        self.d_b.append(delta)
        tmp_db = torch.zeros(self.output_size, 1)
        for i in range(self.output_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        # 第三隐藏层
        # delta = ((self.W[3].T).mm(delta)) * (self.a[2] - self.a[2] * self.a[2])
        delta = ((self.W[3].T).mm(delta)) * self.d_ReLU(self.z[2])
        self.d_W.append(delta.mm(self.a[1].T))
        tmp_db = torch.zeros(self.hidden3_size, 1)
        for i in range(self.hidden3_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        # 第二隐藏层
        # delta = ((self.W[2].T).mm(delta)) * (self.a[1] - self.a[1] * self.a[1])
        delta = ((self.W[2].T).mm(delta)) * self.d_ReLU(self.z[1])
        self.d_W.append(delta.mm(self.a[0].T))
        tmp_db = torch.zeros(self.hidden2_size, 1)
        for i in range(self.hidden2_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        # 第一隐藏层
        # delta = ((self.W[1].T).mm(delta)) * (self.a[0] - self.a[0] * self.a[0])
        delta = ((self.W[1].T).mm(delta)) * self.d_ReLU(self.z[0])
        self.d_W.append(delta.mm((x.T)))
        tmp_db = torch.zeros(self.hidden1_size, 1)
        for i in range(self.hidden1_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        self.d_W.reverse()
        self.d_b.reverse()

    def optimize(self):
        for i in range(4):
            self.W[i] = self.W[i] - learning_rate * self.d_W[i]
            self.b[i] = self.b[i] - learning_rate * self.d_b[i]


def AccuarcyCompute(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    test_np = (np.argmax(pred, 0) == label)
    test_np = np.float32(test_np)
    return np.sum(test_np)


model = MLP()

model.init_parameters()

torch.set_printoptions(threshold=np.inf)

n_iter = 0
for x in range(EPOCH):
    for i, data in enumerate(train_dataset):
        (inputs, labels) = data
        inputs = inputs.view(-1, 28 * 28).T
        t = torch.zeros(10, min(Batch_size, labels.size()[0]))
        for j in range(min(Batch_size, labels.size()[0])):
            t[labels[j]][j] = 1
        y_hat = model.forward_pro(inputs)

        loss = 0
        for j in range(labels.size()[0]):
            loss += model.loss_cal((y_hat.T)[j], labels[j])
        loss = loss / labels.size()[0]

        n_iter += 1
        model.backward_pro(inputs, t)
        model.optimize()
        if (n_iter % 100 == 0):
            print(x, end=" x\n")
            print("loss : %.5f" % loss, end=" ")
            print("train : %.3f" %
                  (AccuarcyCompute(y_hat, labels) / labels.size()[0]),
                  end=" ")

            writer.add_scalar(
                'Train/Accuracy',
                (AccuarcyCompute(y_hat, labels) / labels.size()[0]), n_iter)
            writer.add_scalar('Train/Loss', loss, n_iter)

            accuarcy_list = []
            tmp = 0
            for j, (inputs, labels) in enumerate(test_dataset):
                inputs = inputs.view(-1, 28 * 28).T
                outputs = model.forward_pro(inputs)
                tmp += labels.size()[0]
                accuarcy_list.append(AccuarcyCompute(outputs, labels))
                loss_test = 0
                for k in range(labels.size()[0]):
                    loss += model.loss_cal((outputs.T)[k], labels[k])
                loss = loss / labels.size()[0]
            # (sum(accuarcy_list) / len(accuarcy_list)) 记得 加括号
            writer.add_scalar('Test/Accuracy',
                              sum(accuarcy_list) / tmp, n_iter)
            writer.add_scalar('Test/Loss', loss, n_iter)
            print("test : ",
                  format(sum(accuarcy_list) / tmp, '.3f'),
                  end=" \n")


