import matplotlib.pyplot as plt

# Data for plotting
query_iterations = list(range(1,11))
resnet18_uncertainty = [53, 58, 58, 60, 61, 61, 64, 64, 64, 63, 64, 63, 64, 65, 64, 65, 64, 65, 64, 65]
resnet18_margin = [56, 59, 60, 63, 62, 63, 66, 61, 66, 63, 66, 66, 67, 66, 67, 68, 67, 65, 68, 68]
resnet18_entropy=[58,53,59,62,64,64,65,64,65,65,64,66,66,64,67,67,66,67,68,69]
resnet18_uncertainty_1=[87,91,87,97,95,98,98,98,98,98]
resnet18_margin_1=[87,97,93,95,97,97,98,98,98,98]
resnet18_entropy_1=[78,81,96,92,96,98,96,98,97,98]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(query_iterations, resnet18_uncertainty_1, marker='o', linestyle='-', label='ResNet18-Uncertainty')
plt.plot(query_iterations, resnet18_margin_1, marker='s', linestyle='-', label='ResNet18-Margin')
plt.plot(query_iterations,resnet18_entropy_1,marker='D',linestyle='-',label='ResNet18-Entropy')
plt.xticks(range(0, 11, 2))
# Adding labels and title
plt.xlabel('Query Iteration')
plt.ylabel('Classification Accuracy (%)')
plt.title('Incremental classification accuracy (MNIST)')
plt.ylim(20, 100)
plt.legend()
plt.grid(True)

# Show plot
plt.savefig('res_1')
plt.show()