import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import sklearn


def print_roc_curve(y_test, y_score, fold, n_classes, tprs, aucs):
    lw = 2

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    fpr = {}
    tpr = {}
    roc_auc = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))
        interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)
    lw = 2
    ax.plot(fpr[0], tpr[0], lw=lw, label='ROC fold {}'.format(fold))
    while fold==4:
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="osa 5-flod roc curve")
        ax.legend(loc="lower right")
        plt.show()

# def plot_flod_roc():
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     fig, ax = plt.subplots()
#     for i, (train, test) in enumerate(cv.split(X, y)):
#         classifier.fit(X[train], y[train])
#         viz = plot_roc_curve(classifier, X[test], y[test],
#                              name='ROC fold {}'.format(i),
#                              alpha=0.3, lw=1, ax=ax)
#         interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(viz.roc_auc)
#
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#             label='Chance', alpha=.8)
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#             lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
#
#     ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#            title="Receiver operating characteristic example")
#     ax.legend(loc="lower right")
#     plt.show()


def statics(model, device, test_loader, model_save_path, exp, num_class):
    # model = model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    # with experiment.test():
    predlist = torch.zeros(0, dtype=torch.long).cpu()
    lbllist = torch.zeros(0, dtype=torch.long).cpu()
    correct = 0
    f1 = 0
    y_prob = []
    with torch.no_grad():
        for inputs, labels, _C in tqdm(test_loader):
            data = inputs.to(device)
            data = data.float()
            target = _C.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            probs = torch.nn.functional.softmax(output, dim=1)
            correct += pred.eq(target).cpu().sum().item()
            y_prob.append(probs.detach().cpu().numpy())
            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, target.view(-1).cpu()])
        y_prob = np.concatenate(y_prob)

    pathdir = 'resalut/' + exp
    path = 'resalut/' + exp + '/outputall.txt'
    path_out = 'resalut/' + exp + '/output.txt'
    if os.path.exists(pathdir)==False:
        os.makedirs(pathdir)
    with open(path, 'a') as f:
        conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
        print(conf_mat,file=f)
        print(conf_mat)
        class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
        # print(class_accuracy,file=f)
    conf_matrix = torch.zeros(num_class, num_class)
    for t, p in zip(lbllist, predlist):
        conf_matrix[t, p] += 1

    TP = conf_matrix.diag()
    for c in range(num_class):
        idx = torch.ones(num_class).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[
            idx.nonzero()[:,
            None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))
        sensitivity = TP[c] / (TP[c] + FN)
        specifity = TN / (TN + FP)
        print('sensitivity = {};specifity = {}'.format(sensitivity, specifity))
        # if c == 0:
        with open(path_out, 'a') as ww:
            print(str(sensitivity.item()) + ';' + str(specifity.item()), file=ww)

    label = np.asarray(lbllist)
    pred = np.asarray(predlist)
    # y_pred = np.asarray(y_pred)
    if num_class == 4:
        target_names = ['BS', 'HS', 'snore', 'noise']
    elif num_class == 3:
        target_names = ['BS', 'HS+Snore', 'NOISE', ]

    with open(path, 'a') as f:
        print(classification_report(label, pred, target_names=target_names), file=f)

    y_onehot = np.zeros((label.shape[0], num_class), dtype=np.uint8)
    y_onehot[np.arange(label.shape[0]), label] = 1
    tprs = []
    aucs = []
    # print_roc_curve(y_onehot, y_prob, fold, 2, tprs, aucs)
    # ax = plt.gca()
    return y_onehot, y_prob
    # plot_roc_curve(model, xtest,ytest, ax=ax, alpha=0.8)
    # plt.show()
    # fpr, tpr, thresholds = roc_curve(label, y_pred)
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # plt.show()

def calc(model, device, test_loader,num_class): #計算spesicifity sensitivity
    model.to(device)
    model.eval()
    # with experiment.test():
    predlist = torch.zeros(0, dtype=torch.long).cpu()
    lbllist = torch.zeros(0, dtype=torch.long).cpu()
    correct = 0
    y_prob = []
    f1 = 0
    with torch.no_grad():
        for inputs, labels, _C in tqdm(test_loader):
            data = inputs.to(device)
            #data = data.permute(0, 3, 1, 2)
            data = data.float()
            target = _C.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            probs = torch.nn.functional.softmax(output, dim=1)
            correct += pred.eq(target).cpu().sum().item()
            y_prob.append(probs.detach().cpu().numpy())
            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, target.view(-1).cpu()])
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)
    f1 = sklearn.metrics.f1_score(lbllist, predlist, average='macro')
    print(f1)
    # conf_matrix = torch.zeros(num_class, num_class)
    # for t, p in zip(lbllist, predlist):
    #     conf_matrix[t, p] += 1
    #
    # TP = conf_matrix.diag()
    # for c in range(num_class):
    #     idx = torch.ones(num_class).byte()
    #     idx[c] = 0
    #     # all non-class samples classified as non-class
    #     TN = conf_matrix[
    #         idx.nonzero()[:,
    #         None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
    #     # all non-class samples classified as class
    #     FP = conf_matrix[idx, c].sum()
    #     # all class samples not classified as class
    #     FN = conf_matrix[c, idx].sum()
    #
    #     spe = TN / (TN + FP)
    #     sen = TP[c] / (TP[c] + FN)
    #     f1 += (spe+sen)/2
    return f1


def calc_wavgram(model, device, test_loader,num_class): #計算spesicifity sensitivity
    model.to(device)
    model.eval()
    # with experiment.test():
    predlist = torch.zeros(0, dtype=torch.long).cpu()
    lbllist = torch.zeros(0, dtype=torch.long).cpu()
    correct = 0
    y_prob = []
    f1 = 0
    with torch.no_grad():
        for mel, target in tqdm(test_loader):
            mel = mel.float()
            mel, target = mel.to(device), target.to(device)
            model = model.to(device)
            output = model(mel)
            _, pred = torch.max(output, 1)
            probs = torch.nn.functional.softmax(output, dim=1)
            correct += pred.eq(target).cpu().sum().item()
            y_prob.append(probs.detach().cpu().numpy())
            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, target.view(-1).cpu()])
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)
    f1 = sklearn.metrics.f1_score(lbllist, predlist, average='macro')
    print(f1)
    return f1

def statics_wavgram(model, device, test_loader, model_save_path, exp, num_class):
    # model = model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    # with experiment.test():
    predlist = torch.zeros(0, dtype=torch.long).cpu()
    lbllist = torch.zeros(0, dtype=torch.long).cpu()
    correct = 0
    f1 = 0
    y_prob = []
    with torch.no_grad():
        for mel, target in tqdm(test_loader):
            mel = mel.float()
            mel, target = mel.to(device), target.to(device)
            model = model.to(device)
            output = model(mel)
            _, pred = torch.max(output, 1)
            probs = torch.nn.functional.softmax(output, dim=1)
            correct += pred.eq(target).cpu().sum().item()
            y_prob.append(probs.detach().cpu().numpy())
            predlist = torch.cat([predlist, pred.view(-1).cpu()])
            lbllist = torch.cat([lbllist, target.view(-1).cpu()])
        y_prob = np.concatenate(y_prob)

    pathdir = 'resalut/' + exp
    path = 'resalut/' + exp + '/outputall.txt'
    path_out = 'resalut/' + exp + '/output.txt'
    if os.path.exists(pathdir)==False:
        os.makedirs(pathdir)
    with open(path, 'a') as f:
        conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
        print(conf_mat,file=f)
        print(conf_mat)
        class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
        # print(class_accuracy,file=f)
    conf_matrix = torch.zeros(num_class, num_class)
    for t, p in zip(lbllist, predlist):
        conf_matrix[t, p] += 1

    TP = conf_matrix.diag()
    for c in range(num_class):
        idx = torch.ones(num_class).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[
            idx.nonzero()[:,
            None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))
        sensitivity = TP[c] / (TP[c] + FN)
        specifity = TN / (TN + FP)
        print('sensitivity = {};specifity = {}'.format(sensitivity, specifity))
        # if c == 0:
        with open(path_out, 'a') as ww:
            print(str(sensitivity.item()) + ';' + str(specifity.item()), file=ww)

    label = np.asarray(lbllist)
    pred = np.asarray(predlist)
    # y_pred = np.asarray(y_pred)
    if num_class == 4:
        target_names = ['BS', 'HS', 'snore', 'noise']
    elif num_class == 3:
        target_names = ['BS', 'HS+Snore', 'NOISE', ]

    with open(path, 'a') as f:
        print(classification_report(label, pred, target_names=target_names), file=f)

    y_onehot = np.zeros((label.shape[0], num_class), dtype=np.uint8)
    y_onehot[np.arange(label.shape[0]), label] = 1
    tprs = []
    aucs = []
    # print_roc_curve(y_onehot, y_prob, fold, 2, tprs, aucs)
    # ax = plt.gca()
    return y_onehot, y_prob