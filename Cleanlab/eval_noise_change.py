from cleanlab.pruning import get_noise_indices
import json
import numpy as np
import os
import torchvision.datasets as datasets

def get_rate_change(right_label,noise_labl,pyx,frac_noise=0.5,prune_method="both",logger=None):
    orignal = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    noise_dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    after_prune = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    after_noise = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    correct = []
    find_noise = []
    orignal_noise_id = []

    acc = sum(np.array(right_label) == np.argmax(pyx, axis=1)) / float(len(right_label))
    logger.info('Training Accuracy: {:.25}'.format(acc))

    ordered_label_errors = get_noise_indices(
        s=noise_labl,
        psx=pyx,
        frac_noise=frac_noise,
        prune_method=prune_method,
        sorted_index_method="normalized_margin",  # Orders label errors
    )
    for id in range(len(right_label)):
        orignal[right_label[id]] += 1
        if right_label[id] != noise_labl[id]:
            orignal_noise_id.append(id)
            noise_dic[right_label[id]] += 1
            if id in ordered_label_errors:
                correct.append(id)

        if id in ordered_label_errors:
            find_noise.append(id)
        else:
            after_prune[right_label[id]] += 1
            if right_label[id] != noise_labl[id]:
                after_noise[right_label[id]] += 1

    logger.info("origanl noise rate: {:.3%}".format(len(orignal_noise_id) / 50000))
    score = (len(orignal_noise_id) - len(correct)) / (50000 - len(find_noise))
    logger.info("--> after pruning noise rate：{:.3%}".format(score))
    score = len(correct) / len(orignal_noise_id)
    logger.info('prune real noise/actual noise: {:.3%}'.format(score))
    if len(find_noise) == 0:
        score=0
    else:
        score = len(correct) / len(find_noise)
    logger.info(' prune noise sample  {}'.format(len(find_noise)))
    logger.info('prune real noise/prune sample: {:.2%}'.format(score))
    logger.info("----------------------------------------------")
    for i in range(10):
        logger.info("class: "+str(i)+"    noise rate: {:.3%}".format(noise_dic[i]/orignal[i]))
        logger.info("---> after pruning noise rate: {:.3%}".format(after_noise[i] / after_prune[i]))