import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

name_dic = {
        'wheat':['NOR','F&S','SD','MY','AP','BN','BP', 'IM', 'Recall','Accuracy','F1-score'], 
        'maize':['NOR','F&S','SD','MY','AP','BN','HD', 'IM', 'Recall','Accuracy','F1-score'],
        'sorg':['NOR','F&S','SD','MY','AP','BN','HD',  'IM', 'Recall','Accuracy','F1-score'],
        'rice':['NOR','F&S','SD','MY','AP','BN','UN', 'IM','Recall','Accuracy','F1-score']}

res_dict = {'R1':[], 'R2':[], 'SampleID':[], 'Kappa':[]}
data_path = '/media/dyw/DATA/Data/KappaData'

def read_data(txt_path, sampld_id):
    all_labels = []
    all_files = []
    for line in open(txt_path,'r').readlines():
        line = line.replace('\n','')
        content = line.split(' ')
        if sampld_id in content[0]:
            all_files.append(content[0])
            all_labels.append(int(content[1]))
    return all_files, all_labels 


def get_kappa():

    lst = ['T1', 'T1_S1', 'T1_S2', 'T2', 'T2_S1', 'T2_S2']
    N = len(lst)

    for grain_t in ['wheat','rice','sorg']:
        for ix in range(N-1):
            for iy in range(ix+1,N):
                if lst[ix][:2]!=lst[iy][:2]:
                    continue
                txt1 = os.path.join(data_path, f'{grain_t}_{lst[ix]}.txt') 
                txt2 = os.path.join(data_path, f'{grain_t}_{lst[iy]}.txt') 
                for sample_id in range(1,7):
                    sample_id = f'Sample{sample_id}'
                    files1, labels1 = read_data(txt1, sample_id)
                    files2, labels2 = read_data(txt2, sample_id)
                    assert files1==files2

                    yt = [name_dic[grain_t][ix] for ix in labels1]
                    yp = [name_dic[grain_t][ix] for ix in labels2]

                    C2 = confusion_matrix(yt, yp, labels=name_dic[grain_t])
                    C2 = C2.astype(float)

                    p0 = np.sum(C2[range(7), range(7)])/np.sum(C2)

                    _s = 0
                    for ik in range(7):
                        _s += np.sum(C2[ik,:])*np.sum(C2[:,ik])
                    
                    pe = _s / (np.sum(C2)**2)

                    kappa = (p0-pe)/(1-pe)

                    res_dict['R1'].append(f'{grain_t}_{lst[ix]}')
                    res_dict['R2'].append(f'{grain_t}_{lst[iy]}')
                    res_dict['SampleID'].append(sample_id)
                    res_dict['Kappa'].append(kappa)

                    desc = f'R1:{grain_t}_{lst[ix]},R2:{grain_t}_{lst[iy]},SampleID:{sample_id},Kappa:{kappa:.3f}'
                    print(desc)

                    fig,ax = plt.subplots()
                    recall = [C2[ix][ix]/(sum(C2[ix,:])+1e-5) for ix in range(8)]
                    precision = [C2[ix][ix]/(sum(C2[:,ix])+1e-5) for ix in range(8)]
                    f1 = [2*recall[ix]*precision[ix]/(precision[ix]+recall[ix]+1e-6) for ix in range(8)]

                    f1 = [round(f1[ix]*1000)/10 for ix in range(8)]
                    recall = [round(recall[ix]*1000)/10 for ix in range(8)]
                    precision = [round(precision[ix]*1000)/10 for ix in range(8)]
                    # acc = [round(acc[ix]*1000)/10 for ix in range(8)]

                    C2[:,8][:8] = recall
                    C2[:,9][:8] = precision
                    C2[:,10][:8] = f1
                    C2 = C2[:8]
                    df=pd.DataFrame(C2,index=name_dic[grain_t][:8],columns=name_dic[grain_t])
                    sns.set(font_scale=1.2)
                    plt.rc('font',size=10)
                    sns.heatmap(df, annot=True, cbar=True, fmt='.5g', cmap="GnBu",  vmin=0, vmax=400, center=200, square=True, linewidths=1, linecolor='black', cbar_kws={"shrink": 0.8}) #cmap="Reds", # annot_kws={'color':'black'},
                    
                    ax.tick_params(right=False, top=False, labelright=False, labelbottom=False, labelrotation=90, labeltop=True, labelleft=True) # labelrotation=45

                    ax.set_title(f'Kappa={kappa:.3f}')

                    plt.yticks(rotation=0)

                    # ax.set_title('confusion matrix') #标题
                    # ax.set_xlabel(f'R50 on {grain_t} data') #x 轴
                    # ax.set_ylabel('true') #y 轴
                    fig.savefig(f'./kappas/{desc}_matrix.png',dpi=300)

                    plt.close()


    df = pd.DataFrame(res_dict)
    df.to_csv('kappa_res.csv')


get_kappa()