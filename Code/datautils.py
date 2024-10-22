"""
@author: iopenzd
"""
import numpy as np
from scipy.io import arff


def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_UCR(Path, dataset):
    train_path = Path + '/' + dataset + '_TRAIN.arff'
    test_path = Path + '/' + dataset + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)  # data是数据，meta是类别和属性信息
        f.close()

    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])  # 创建了字典，每一个类别属性对应的rawlabel就是类别的值
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))  # 对变长补0
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        X_train = np.array(TRAIN_DATA)
        y_train = np.array(TRAIN_LABEL)
        X_test = np.array(TEST_DATA)
        y_test = np.array(TEST_LABEL)
    return X_train, y_train, X_test, y_test
