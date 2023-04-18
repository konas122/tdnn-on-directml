import torch
import numpy
import random
import amd.amd_audio as audio
import amd.amd_loader as loader
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt


def tuneThresholdfromScore(scores, labels, target_fa):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('../img/ROC.jpg', dpi=400)

    prec, recall, _ = metrics.precision_recall_curve(labels, scores, pos_label=1)
    metrics.PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.savefig('../img/PR.jpg', dpi=400)
    # F1 = (2 * prec * recall

    fnr = 1 - tpr
    tunedThreshold = []

    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE])

    return tunedThreshold[1][0], eer, auc, fpr, fnr


def ComputeErrorRates(scores, labels, threshold=0.9655, p=0.01):
    assert len(scores) == len(labels), f'Error: {scores} {labels}\n'
    predict = []
    threshold = threshold if 0.96 <= threshold < 0.99 else 0.96
    for i in range(len(scores)):
        if scores[i] > threshold:
            predict.append(1)
        else:
            predict.append(0)
    matrix = metrics.confusion_matrix(labels, predict)
    [TN, FP], [FN, TP] = matrix
    matrix = numpy.array([[TP, FN], [FP, TN]])

    metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['True', 'False']).plot()
    plt.savefig('../img/confusion_matrix.jpg', dpi=400)

    FAR = FP / (FP + TN)
    FRR = FN / (TP + FN)
    minDCF = FAR * (1 - p) + FRR * p
    return matrix, minDCF


def get_embedding(net, name, device):
    net.aug = False
    net.embedding = True
    # net.to(device)
    wav = audio.loadWAV(filename=name)
    wav = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = net(wav)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def dic_process(dic):
    result = {}
    value = list(dic.values())[0]
    embedding_list = []
    for item in dic.items():
        if item[1] != value:
            value = item[1]
            embedding_list = []
        embedding_list.append(item[0])
        result[value] = embedding_list
    return result


def eval_net(net, device, folder_num=-1, file_num=-1):
    labels = []
    embed_dict = {}
    score_list = []
    enroll, test, folder_num = loader.load_files("test", folder_num, file_num, 9)
    enroll = dic_process(enroll)
    test = dic_process(test)
    net.eval()
    net.to(device)

    for key in enroll:
        count = 0
        embed = None
        for name in enroll[key]:
            if count >= len(enroll[key]):
                break
            count += 1
            embedding = get_embedding(net, name, device)

            if count == 1:
                embed = embedding
            else:
                embed = torch.cat([embed, embedding])
        embed = torch.mean(embed, dim=0).unsqueeze(0)
        embed_dict[key] = embed

    for item in enroll:
        dict_key_ls = list(enroll.keys())
        random.shuffle(dict_key_ls)
        for label in dict_key_ls:
            if label == item:
                y_true = 1
            else:
                y_true = 0
            num = random.randint(0, len(test[label]) - 1)
            embed1 = get_embedding(net, test[label][num], device)
            embedding = embed_dict[item]

            score = torch.matmul(embed1, embedding.mT).cpu().numpy().reshape(-1)
            score_list.append(score)
            labels.append(y_true)

    threshold, EER, AUC, _, _ = tuneThresholdfromScore(score_list, labels, [1, 0.1])
    _, minDCF = ComputeErrorRates(score_list, labels, threshold)
    return EER, minDCF


if __name__ == '__main__':
    train_dict, test_dict, number = loader.load_files("train", 40, 20, 1.5)
    dic_process(train_dict)
    # print(train_dict)

    # embed = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4],
    #                           [0.5, 0.6, 0.7, 0.8]])
    # sum = torch.matmul(embed, embed.T)
    # sum = torch.sum(sum, dim=[0, 1], keepdim=False)
    # print(sum)
