from matplotlib.pylab import plt
import numpy
import pandas

def f1_score(ground, pred, label):
    tp = numpy.sum((ground==label) & (pred==label))
    fp = numpy.sum((ground!=label) & (pred==label))
    fn = numpy.sum((pred!=label) & (ground==label))
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def f1_macro(ground, pred):
    return numpy.mean([f1_score(ground, pred, label) for label in numpy.unique(ground)])

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
    