import numpy as np
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



def tSNE(labels, preds, embds1, embds2, embds3):
    
    reduced_embds1 = TSNE(n_components=2, init='pca').fit_transform(embds1)
    reduced_embds2 = TSNE(n_components=2, init='pca').fit_transform(embds2)
    reduced_embds3 = TSNE(n_components=2, init='pca').fit_transform(embds3)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=200)

    colors = ['indianred', 'darkorange', 'darkseagreen',
              'steelblue', 'mediumpurple', 'gold']
    Label_Com = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    for index in range(6):
        positions = np.where(labels == index)
        ax1.scatter(reduced_embds1[positions, 0], reduced_embds1[positions, 1],
                    c=colors[index], s=5, alpha=0.7, marker='8', linewidth=0)
        ax2.scatter(reduced_embds2[positions, 0], reduced_embds2[positions, 1],
                    c=colors[index], s=5, alpha=0.7, marker='8', linewidth=0)
        ax3.scatter(reduced_embds3[positions, 0], reduced_embds2[positions, 1],
                    c=colors[index], s=5, alpha=0.7, marker='8', linewidth=0)

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(30)
    # added this to get the legend to work
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels=Label_Com, loc='upper right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.title.set_text('No encoder')

    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(30)
    # added this to get the legend to work
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels=Label_Com, loc='upper right')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.title.set_text('sequence encoder')
    
    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(30)
    # added this to get the legend to work
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels=Label_Com, loc='upper right')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.title.set_text('sequence + graph encoders')
    
    
def confusion(labels, preds):
    C = confusion_matrix(labels,preds)
    Name = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
      
    plt.figure()
    sns.set()
    sns.heatmap(C,annot=True,fmt="d",cmap='GnBu',annot_kws={"fontsize":8}) 
    plt.title('confusion matrix') 
    plt.xlabel('prediction') 
    plt.ylabel('true') 
    num_local = np.array(range(len(Name)))    
    plt.xticks(num_local,Name,rotation='45')    
    plt.yticks(num_local,Name,rotation='45')
    plt.show()

    
embds1 = np.load('./data4visualize/embds_1.npy')
embds2 = np.load('./data4visualize/embds_2.npy')
embds3 = np.load('./data4visualize/embds_3.npy')
labels = np.load('./data4visualize/label.npy')
preds = np.load('./data4visualize/pred.npy')

tSNE(labels, preds, embds1, embds2, embds3)

confusion(labels, preds)

t = classification_report(labels, preds, target_names=['hap', 'sad', 'neu', 'ang', 'exc', 'fru'])
print(t)