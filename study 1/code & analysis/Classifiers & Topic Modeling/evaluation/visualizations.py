import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(conf_matrix, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()
