from matplotlib.pylab import plt
import numpy
import pandas

def f1_macro(ground, pred):
    labels = numpy.unique(ground)
    f1 = 0
    for label in labels:
        true_positive = numpy.sum((ground==label) & (pred==label))
        false_positive = numpy.sum((ground!=label) & (pred==label))
        false_negative = numpy.sum((pred!=label) & (ground==label))
    
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        
        f1 += 2 * (precision * recall) / (precision + recall)
        
    return f1 / len(labels)
        

def accuracy(ground, pred):
    return (ground == pred).sum() / len(ground)


def confusion_matrix(ground, pred):
    ground_df = pandas.Series(ground, name="Ground")
    pred_df = pandas.Series(pred, name="Predicted")
    
    return pandas.crosstab(ground_df, pred_df)
    
    
def plot_loss(training_loss: list[float], validation_loss: list[float], num_epoch: int, savefig_path: str):
    plt.title('Training and Validation Loss')
    
    plt.plot(range(1, num_epoch+1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epoch+1), validation_loss, label='Validation Loss')
    
    plt.legend(loc='best')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.savefig(savefig_path)

    try:
        plt.show()
    except UserWarning as error:
        print(f"{error}\nUnable to show the Loss Graph interactivaly")
    
def plot_f1(f1_scores: list[float], num_epoch: int, savefig_path: str):
    plt.title('F1 Score')
    
    plt.plot(range(1, num_epoch+1), f1_scores, label='F1 Scores')
    
    plt.legend(loc='best')
    
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    
    plt.savefig(savefig_path)

    try:
        plt.show()
    except UserWarning as error:
        print(f"{error}\nUnable to show the Loss Graph interactivaly")