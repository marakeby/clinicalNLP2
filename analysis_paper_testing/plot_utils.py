# __author__ = haitham
# elmarakeby
from sklearn import metrics
from sklearn.metrics import average_precision_score
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt, ticker
import numpy as np
import itertools


def plot_prc(ax, y_test, y_pred_score, save_dir, color, fontproperties, label=''):
    # plt.figure(fig.number)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_score)
    roc_auc = average_precision_score(y_test, y_pred_score)
    #     plt.plot(recall, precision, label=label + '(area= %0.2f)' % roc_auc, linewidth=2, color=color)
    ax.plot(recall, precision, label=label + '(area= %0.2f)' % roc_auc, linewidth=2, color=color)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall',fontproperties )
    ax.set_ylabel('Precision', fontproperties)

def plot_roc(ax, y_test, y_pred_score, save_dir,color, fontproperties, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc, linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)


def plot_confusion_matrix(ax, cm, classes, fontproperties, labels=None,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fontsize= fontproperties['size']
    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.tick_params(axis=u'both', which=u'both', length=0)
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))
    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label',  fontproperties )
    ax.set_xlabel('Predicted label', fontproperties)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontproperties)
    ax.set_yticks([t-0.25 for t in tick_marks])
    ax.set_yticklabels(classes, fontproperties, rotation=90)