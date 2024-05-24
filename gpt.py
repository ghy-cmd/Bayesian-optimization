import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner,BayesianOptimizer
from modAL.uncertainty import uncertainty_sampling,classifier_uncertainty
from skorch import NeuralNetClassifier
import numpy as np
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据转换
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

# 加载 CIFAR-10 数据集
cifar10_train = datasets.CIFAR10(root='/hy-tmp', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root='/hy-tmp', train=False, download=True, transform=transform)

# 定义初始训练集大小
initial_training_size = 10000

# 划分初始训练集和未标记数据池
initial_training_indices, pool_indices = train_test_split(range(len(cifar10_train)), train_size=initial_training_size, random_state=42)
initial_training_set = Subset(cifar10_train, initial_training_indices)
pool_set = Subset(cifar10_train, pool_indices)

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
    max_epochs=5
)

# 初始化 ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial.numpy(),
    y_training=y_initial.numpy(),
    # query_strategy=uncertainty_sampling
)

X_test, y_test = load_data(cifar10_test)
# 设置目标准确率阈值和不确定度阈值
target_accuracy = 0.70
uncertainty_threshold = 0.4

# 主动学习循环
# 开始主动学习
performance_history = []
n_instances = 1000  # 每次选择的样本数
i=0
with open('performance_history_{}_{}_{}.log'.format(target_accuracy,uncertainty_threshold,n_instances), 'w') as log_file:    
    while learner.score(X_test.numpy(), y_test.numpy()) < target_accuracy and i<20:
        # query_idx, query_instance = learner.query(X_pool.numpy(),n_instances=2000)
        # stream_idx = np.random.choice(range(len(X_pool)))
        # X_stream = X_pool[stream_idx].unsqueeze(0).numpy()
        # y_stream = np.array([y_pool[stream_idx].item()])
        i=i+1
        stream_indices = np.random.choice(range(len(X_pool)), size=n_instances, replace=False)
        X_stream = X_pool[stream_indices].numpy()
        y_stream = np.array([y_pool[idx].item() for idx in stream_indices])

        # uncertainties = [classifier_uncertainty(learner, X_stream[i]) for i in range(n_instances)]
        uncertainties = classifier_uncertainty(learner, X_stream)
        to_teach = [i for i in range(n_instances) if uncertainties[i] >= uncertainty_threshold]

        # if classifier_uncertainty(learner, X_stream) >= uncertainty_threshold:
        #     learner.teach(X=X_stream, y=y_stream)
        #     # X_pool = torch.tensor(np.delete(X_pool.numpy(), query_idx, axis=0))
        #     # y_pool = torch.tensor(np.delete(y_pool.numpy(), query_idx, axis=0))
        #     new_score = learner.score(X_test.numpy(), y_test.numpy())
        #     performance_history.append(new_score)
        #     log_file.write(f'样本 被查询，新的准确率: {new_score:.2f}\n')
        #     print(f'样本 被查询，新的准确率: {new_score:.2f}')
        while len(to_teach) < 2 and uncertainty_threshold > 0:
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
with open('performance_history_{}_{}_{}.log'.format(target_accuracy,uncertainty_threshold,n_instances), 'a') as log_file:
    log_file.write(f'最终测试集准确率: {final_accuracy * 100:.2f}%\n')

# 绘制 performance_history
plt.figure(figsize=(10, 6))
plt.plot(performance_history, marker='o')
plt.title('Model Performance over Time')
plt.xlabel('Number of Queries')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('performance_history_{}_{}_{}.png'.format(target_accuracy,uncertainty_threshold,n_instances))
plt.show()