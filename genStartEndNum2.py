import numpy as np
import random
import copy
import statistics as stat
from statistics import NormalDist
import matplotlib.pyplot as plt
from pylab import rcParams
import scipy
from numpy import inf
import scipy.stats as st

def gaussian_algorithm(classNum,class_list,valList,maxNum):
    sample_matrix = np.array(valList)
    k = 0
    true_means = []
    null_means = []
    while k < 30:
        # get the half random sample and half random variables list
        half_rand_matrix,sample_ind_list = selectHalfRandom(sample_matrix)
        half_rand_class_list = []
        for z in sample_ind_list:
            half_rand_class_list.append(class_list[z])
        classNum_list = []
        for i in range(classNum):
            classNum_list.append(i + 1)
        # create true distribution with the original class number
        true_dist_classNum = copy.deepcopy(half_rand_class_list)
        # create null distribution with the randomly assigned class number
        null_dist_classNum = []
        for i in range(len(half_rand_class_list)):
            null_dist_classNum.append(random.choice(classNum_list))
        # calculate the tru and null fisher ratio
        true_fisherRatio = cal_fish_ratio(half_rand_matrix, true_dist_classNum, classNum)
        null_fisherRatio = cal_fish_ratio(half_rand_matrix, null_dist_classNum, classNum)
        true_mean_fisher_ratio = np.mean(true_fisherRatio)
        null_mean_fisher_ratio = np.mean(null_fisherRatio)
        true_means.append(true_mean_fisher_ratio)
        null_means.append(null_mean_fisher_ratio)
        k = k + 1
    for i in range(len(true_means)):
        if true_means[i] == np.inf:
            true_means[i] = 0
    for i in range(len(null_means)):
        if null_means[i] == np.inf:
            null_means[i] = 0
    true_fisher_mean = np.mean(true_means)
    true_fisher_std = np.std(true_means)
    null_fisher_mean = np.mean(null_means)
    null_fisher_std = np.std(null_means)
    ####################################  END GRAPH CODE ###################################
    startNum = NormalDist(mu=true_fisher_mean, sigma=true_fisher_std).inv_cdf(0.95)
    endNum = NormalDist(mu=null_fisher_mean, sigma=null_fisher_std).inv_cdf(0.05)

    ####################################  START GRAPH CODE ###################################
    # generate a histogram by using mean
    x_values = np.arange(-5, 15, 0.1)
    true_y_values = scipy.stats.norm(true_fisher_mean, true_fisher_std)
    plt.plot(x_values, true_y_values.pdf(x_values), label= 'true gaussian')
    null_y_values = scipy.stats.norm(null_fisher_mean, null_fisher_std)
    plt.plot(x_values, null_y_values.pdf(x_values), label='null gaussian')
    rcParams['figure.figsize'] = 10, 10


    #replace the inf in mean with the largest number in original matrix

    plt.hist(true_means, density=True,color="#3468eb",alpha=.6, label="true fisher mean")
    plt.hist(null_means, density=True,color="#34ebba",alpha=.6, label=" null fisher mean")
    plt.ylabel("Probability")
    plt.xlabel("Mean")
    plt.tight_layout()
    mn, mx = plt.xlim()
    plt.plot([startNum,startNum],[0,0.5],color = 'blue', label="Start Number")
    plt.plot([endNum,endNum],[0,0.5],color = 'red', label="End Number")
    plt.plot(startNum, 0.5, color='blue')
    plt.plot(endNum,  0.5, color='red')
    plt.legend(loc='best')
    plt.xlim(mn, mx)
    plt.savefig('output/FisherMean.png')
    plt.figure().clear()


    return startNum, endNum


# randomly get half sample, half variable matrix
def selectHalfRandom(sample_list):
    idx_list = []
    rand_sample_list = []
    for i in range(len(sample_list)):
        idx_list.append(i)

    total_num = len(sample_list)
    half_num = total_num//2

    rand_idx_list = random.sample(list(idx_list),half_num)
    for idx in rand_idx_list:
        rand_sample_list.append(sample_list[idx])

    return rand_sample_list,rand_idx_list

def cal_fish_ratio(sample_list,class_list,classNum):
    # define a fisher ratio list for all columns with default value 0
    fish_ratio = []
    # for each column sample type we calculate one fisher ratio for one column
    for i in range(len(sample_list[0])):
        #define a data list for all class
        # define a data list contain different class data list
        class_data = []
        for k in range(classNum + 1):
            class_data.append([])
        #for each row of data
        all_data = [row[i] for row in sample_list]
        for ind in range(len(all_data)):
            class_data[int(class_list[ind])].append(all_data[ind])
        class_data = [x for x in class_data if x != []]
        # Here we calculate the fisher ratio for that column
        # calculate the first lumda sqr
        all_data_mean = np.mean(all_data)
        lumdaTop1 = 0
        for z in range(len(class_data)):
            class_data_mean = np.mean(class_data[z])
            lumdaTop1 = lumdaTop1 + (((class_data_mean - all_data_mean)**2)*len(class_data[z]))
        lumdaBottom1 = classNum-1
        lumda1 = lumdaTop1/lumdaBottom1
        lumdaTop2_1 = 0
        for n in range(len(class_data)):
            for j in class_data[n]:
                lumdaTop2_1 = lumdaTop2_1 + (j - all_data_mean)**2
        lumdaTop2_2 = 0
        for p in range(len(class_data)):
            class_data_mean = 0
            for data in class_data[p]:
                class_data_mean = class_data_mean + data
            class_data_mean = class_data_mean/len(class_data[p])
            lumdaTop2_2 = lumdaTop2_2 + (((class_data_mean - all_data_mean) ** 2) * len(class_data[p]))
        lumdaBottom2 = len(all_data) - classNum
        lumda2 = (lumdaTop2_1-lumdaTop2_2)/lumdaBottom2
        fisher_ratio = lumda1/lumda2
        fish_ratio.append(fisher_ratio)
    fish_ratio = np.nan_to_num(fish_ratio, nan=(10 ** -12))
    return fish_ratio