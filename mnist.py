import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner, BayesianOptimizer
from modAL.uncertainty import uncertainty_sampling, classifier_uncertainty,classifier_entropy,classifier_margin
from skorch import NeuralNetClassifier
import numpy as np
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
mnist_train = datasets.MNIST(root='/root/code', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='/root/code', train=False, download=True, transform=transform)

# 定义初始训练集大小
initial_training_size = 5000

# 划分初始训练集和未标记数据池
initial_training_indices, pool_indices = train_test_split(range(len(mnist_train)), train_size=initial_training_size, random_state=42)
initial_training_set = Subset(mnist_train, initial_training_indices)
pool_set = Subset(mnist_train, pool_indices)

# 定义数据加载函数
def load_data(subset):
    loader = DataLoader(subset, batch_size=len(subset), shuffle=True)
    for data in loader:
        return data[0], data[1]

X_initial, y_initial = load_data(initial_training_set)
X_pool, y_pool = load_data(pool_set)

# 定义 ResNet18 模型
class ResNet18Wrapper(torch.nn.Module):
    def __init__(self):
        super(ResNet18Wrapper, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 10)
        
    def forward(self, x):
        return self.resnet(x)

model = ResNet18Wrapper()

# 使用 skorch 包装器
classifier = NeuralNetClassifier(
    module=model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=None,
    verbose=1,
    device=device,
    max_epochs=2
)

# 初始化 ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial.numpy(),
    y_training=y_initial.numpy(),
    # query_strategy=uncertainty_sampling
)

X_test, y_test = load_data(mnist_test)

# 设置目标准确率阈值和不确定度阈值
target_accuracy = 1
uncertainty_threshold = 0.2

# 主动学习循环
performance_history = []
n_instances = 1000  # 每次选择的样本数
i = 0
with open('performance_history_mnist_margin_{}_{}_{}.log'.format(target_accuracy, uncertainty_threshold, n_instances), 'w') as log_file:
    while learner.score(X_test.numpy(), y_test.numpy()) < target_accuracy and i < 10:
        ncertainty_threshold=0.4
        i += 1
        stream_indices = np.random.choice(range(len(X_pool)), size=n_instances, replace=False)
        X_stream = X_pool[stream_indices].numpy()
        y_stream = np.array([y_pool[idx].item() for idx in stream_indices])

        uncertainties = classifier_margin(learner, X_stream)
        to_teach = [i for i in range(n_instances) if uncertainties[i] >= uncertainty_threshold]

        while len(to_teach) < 2:
            uncertainty_threshold -= 0.05
            to_teach = [i for i in range(n_instances) if uncertainties[i] >= uncertainty_threshold]

        if to_teach:
            X_teach = X_stream[to_teach]
            y_teach = y_stream[to_teach]
            learner.teach(X=X_teach, y=y_teach)
            new_score = learner.score(X_test.numpy(), y_test.numpy())
            performance_history.append(new_score)
            log_file.write(f'样本 {stream_indices[to_teach]} 被查询，新的准确率: {new_score:.2f}\n')
            print(f'样本 {stream_indices[to_teach]} 被查询，新的准确率: {new_score:.2f}')
        
            # 移除已查询样本
            X_pool = torch.tensor(np.delete(X_pool.numpy(), stream_indices[to_teach], axis=0))
            y_pool = torch.tensor(np.delete(y_pool.numpy(), stream_indices[to_teach], axis=0))

# 最终测试模型
final_accuracy = learner.score(X_test.numpy(), y_test.numpy())
print(f'最终测试集准确率: {final_accuracy * 100:.2f}%')

# 保存准确率信息到文件
with open('performance_history_mnist_margin_{}_{}_{}.log'.format(target_accuracy, uncertainty_threshold, n_instances), 'a') as log_file:
    log_file.write(f'最终测试集准确率: {final_accuracy * 100:.2f}%\n')

# 绘制 performance_history
plt.figure(figsize=(10, 6))
plt.plot(performance_history, marker='o')
for i, value in enumerate(performance_history):
    plt.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0, 10), ha='center')
# plt.title('Model Performance')
plt.xlabel('Number of Queries')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('performance_history_mnist_margin_{}_{}_{}.png'.format(target_accuracy, uncertainty_threshold, n_instances))
plt.show()
