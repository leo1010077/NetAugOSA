import os
import numpy as np
import csv
dir = './result/'
dest = './results.csv'
print(os.listdir(dir))
with open(dest, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for fold in os.listdir(dir):
        fold_dir = dir + fold + '/'
        f1_txt = fold_dir + 'f1.txt'

        f1 = 0
        complete = True
        with open(f1_txt) as file:

            ls = file.readlines()
            num = len(ls)
            for l in ls:
                f1 += float(l)
        f1 /= num
        output_txt = fold_dir + 'output.txt'
        confusion_m = np.zeros(8)

        with open(output_txt) as file:
            bias = 0
            ls = file.readlines()
            #print(ls)
            for ind, l in enumerate(ls):
                if '---' in l:
                    continue
                confusion_m[bias] += float(l.split(';')[0])
                confusion_m[4 + bias] += float(l.split(';')[1].split('\\')[0])
                #print(bias, ind)
                bias += 1
                if bias == 4:
                    bias = 0
        confusion_m /= num
        if num != 5:
            complete = False
        line = [fold, confusion_m[0], confusion_m[1], confusion_m[2], confusion_m[3], confusion_m[4], confusion_m[5], confusion_m[6], confusion_m[7], f1, complete]
        print(line)
        writer.writerow(line)
    #print(fold, f1)