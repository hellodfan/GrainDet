import numpy as np
import os
from sklearn.metrics import confusion_matrix
from utils.util import plot

from openTSNE import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import time
from multiprocessing import Pool

name_dic = {
        'wheat':['NOR','F&S','SD','MY','AP','BN','BP', 'IM', 'Recall','Accuracy','F1-score'], 
        'maize':['NOR','F&S','SD','MY','AP','BN','HD', 'IM', 'Recall','Accuracy','F1-score'],
        'sorg':['NOR','F&S','SD','MY','AP','BN','HD',  'IM', 'Recall','Accuracy','F1-score'],
        'rice':['NOR','F&S','SD','MY','AP','BN','UN', 'IM','Recall','Accuracy','F1-score']}

def plot_res(fns):
    
    t_start = time.time()
    print("开始执行,进程号为%d" % (os.getpid()))

    for fn in fns:

        all_label = []
        embeds = []
        files = []
        all_pred = []

        for line in open(fn,'r').readlines():
            line = line.replace('\n','')
            content = line.split(' ')
            files.append(content[0])
            all_label.append(int(content[1]))
            embeds.append(list(map(float, content[2:])))
        
        fn = fn.split('/')[-1]
        grain_t = fn.split('_')[1]

        embeds = np.vstack(embeds)
        all_label = np.hstack(all_label)
        all_pred = np.argmax(embeds, axis=1) 

        files = np.hstack(files).tolist()
        tsne_xy = TSNE().fit(embeds)

        label_name_dic = {ix:el for ix, el in enumerate(name_dic[grain_t][:8])}
        fig,ax = plt.subplots(figsize=(8, 5))
        tsne_label = [label_name_dic[el] for el in all_label] 

        ax.set_xlim(xmin=-90,xmax=90)
        ax.set_ylim(ymin=-80,ymax=80)

        plot(tsne_xy, tsne_label, ax, colors=None, label_order=name_dic[grain_t][:8])

        fig.savefig(f'./results/tsne_{fn[:-4]}.jpg', dpi=300)


        sns.set()
        fig,ax = plt.subplots()
        yt = [name_dic[grain_t][ix] for ix in all_label]
        yp = [name_dic[grain_t][ix] for ix in all_pred]

        C2 = confusion_matrix(yt, yp, labels=name_dic[grain_t])
        C2 = C2.astype(np.float)
        recall = [C2[ix][ix]/(sum(C2[ix,:])+1e-6) for ix in range(8)]
        precision = [C2[ix][ix]/(sum(C2[:,ix])+1e-6) for ix in range(8)]
        f1 = [2*recall[ix]*precision[ix]/(precision[ix]+recall[ix]+1e-6) for ix in range(8)]

        f1 = [round(f1[ix]*1000)/10 for ix in range(8)]
        recall = [round(recall[ix]*1000)/10 for ix in range(8)]
        precision = [round(precision[ix]*1000)/10 for ix in range(8)]

        m_f1  = sum(f1)/len(f1)
        m_rec = sum(recall)/len(recall)
        m_pre = sum(precision)/len(precision)
        print(f'{fn}\t precision:{m_pre}\t f1:{m_f1}\t recall:{m_rec}')

        C2[:,8][:8] = recall
        C2[:,9][:8] = precision
        C2[:,10][:8] = f1
        C2 = C2[:8]
        df=pd.DataFrame(C2,index=name_dic[grain_t][:8],columns=name_dic[grain_t])
        sns.set(font_scale=1.2)
        plt.rc('font',size=10)
        sns.heatmap(df, annot=True, cbar=True, fmt='.5g', cmap="GnBu",  vmin=0, vmax=400, center=200, square=True, linewidths=1, linecolor='black', cbar_kws={"shrink": 0.8}) #cmap="Reds", # annot_kws={'color':'black'},
        
        ax.tick_params(right=False, top=False, labelright=False, labelbottom=False, labelrotation=90, labeltop=True, labelleft=True) # labelrotation=45

        plt.yticks(rotation=0)

        # ax.set_title('confusion matrix') #标题
        # ax.set_xlabel(f'R50 on {grain_t} data') #x 轴
        # ax.set_ylabel('true') #y 轴
        fig.savefig(f'./results/{fn[:-4]}_matrix.png',dpi=300)

    t_stop = time.time()
    print(os.getpid(), "执行完毕，耗时%0.2f" % (t_stop-t_start))


if __name__ == "__main__":

    prob_path = './results/svm'

    all_res = []
    fn_lst = os.listdir(prob_path)
    fn_lst = [os.path.join(prob_path, el) for el in fn_lst]
    
    nw = len(fn_lst)
    po = Pool(min(nw,8))

    for i in range(nw):
        po.apply_async(plot_res, ([fn_lst[i]],))

    print("----start----")
    po.close()
    po.join()
    print("-----end-----")
