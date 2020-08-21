import json
import numpy as np
righ_label = json.load(open("/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/labled.json", "r"))
noise_label = json.load(open("/data/glusterfs_cv_04/11121171/AAAI_NL/Baseline_classification/classification_noise/workspace/MobileNet_v2-cifar10-Seed1111/0.4_sym.json", "r"))
# rate = np.array((righ_label==noise_label))
# riggt = [1 if righ_label[i]==noise_label[i] else 0 for i in range(len(righ_label))]
# rate = 1-np.array(riggt).sum()/len(righ_label)
# print(rate)
list = np.zeros(10)
for i in righ_label:
    list[i] +=1

print(list.sum())