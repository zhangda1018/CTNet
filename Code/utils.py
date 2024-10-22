# -*- coding: utf-8 -*-
"""
@author: iopenzd
"""
import warnings, pickle, torch, math, os, random, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import cls_transformer_class

warnings.filterwarnings("ignore")


# loading optimized hyperparameters
def get_optimized_hyperparameters(dataset):
    path = './hyperparameters.pkl'
    with open(path, 'rb') as handle:
        all_datasets = pickle.load(handle)
        if dataset in all_datasets:
            prop = all_datasets[dataset]
    return prop


# 加载用户指定的超参数
def get_user_specified_hyperparameters(args):
    prop = {}
    prop['batch'], prop['lr'], prop['nlayers'], prop['emb_size'], prop['nhead'], prop['task_rate'], prop[
        'masking_ratio'], prop['task_type'] = \
        args.batch, args.lr, args.nlayers, args.emb_size, args.nhead, args.task_rate, args.masking_ratio, args.task_type
    return prop


# loading fixed hyperparameters   加载固定超参数
def get_fixed_hyperparameters(prop, args):
    prop['lamb'], prop['epochs'], prop['ratio_highest_attention'], prop[
        'avg'] = args.lamb, args.epochs, args.ratio_highest_attention, args.avg
    prop['dropout'], prop['nhid'], prop['nhid_task'], prop['nhid_re'], prop['dataset'], prop[
        'Path'] = args.dropout, args.nhid, args.nhid_task, args.nhid_re, args.dataset, args.Path
    return prop


def get_prop(args):
    # loading optimized hyperparameters
    # prop = get_optimized_hyperparameters(args.dataset)

    prop = get_user_specified_hyperparameters(args)  # 加载用户指定的超参数
    prop = get_fixed_hyperparameters(prop, args)  # 加载固定超参数
    return prop


def make_perfect_batch(X, num_inst, num_samples):  # 将输入数组X扩展为具有期望批量大小的完美批量
    extension = np.zeros((num_samples - num_inst, X.shape[1], X.shape[2]))
    X = np.concatenate((X, extension), axis=0)
    return X


def mean_standardize_fit(X):  # 计算均值和标准差
    m1 = np.mean(X, axis=1)  # 计算X数组沿着第一维度（即行）的均值，使用np.mean(X, axis=1)。这将得到一个形状为(X.shape[0],)的数组，其中每个元素是相应行的均值
    mean = np.mean(m1, axis=0)  # 计算步骤1中得到的均值数组的平均值，使用np.mean(m1, axis=0)。这将得到一个标量值，表示所有行的均值的平均值，即整个数组X的均值。

    s1 = np.std(X, axis=1)  # 计算X数组沿着第一维度（即行）的标准差，使用np.std(X, axis=1)。这将得到一个形状为(X.shape[0],)的数组，其中每个元素是相应行的标准差。
    std = np.mean(s1, axis=0)  # 计算步骤3中得到的标准差数组的平均值，使用np.mean(s1, axis=0)。这将得到一个标量值，表示所有行的标准差的平均值，即整个数组X的标准差。
    return mean, std


def mean_standardize_transform(X, mean, std):  # 均值标准化转换
    return (X - mean) / std


def preprocess(prop, X_train, y_train, X_test, y_test):  # 数据训练前处理
    mean, std = mean_standardize_fit(X_train)  # 计算均值和标准差
    # 均值标准化转换
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0]  # 样本数量
    # 根据给定的批量大小，将训练集和测试集的样本数量调整为能够完整处理批量的整数倍，以确保在训练和测试过程中不会丢失任何样本
    num_train_samples = math.ceil(num_train_inst / prop['batch']) * prop['batch']  # math.ceil()函数向上取整
    num_test_samples = math.ceil(num_test_inst / prop['batch']) * prop['batch']

    # 如果输入的样本数量少于期望的批量大小，将使用全零样本进行扩展，以填充不足的部分。
    # 这样可以保证在批量处理数据时每个批量都具有相同的样本数量，方便进行并行计算等操作。
    X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples)
    X_test = make_perfect_batch(X_test, num_test_inst, num_test_samples)

    X_train_cls = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()
    # 将给定的X_train和X_test数组转换为PyTorch张量（Tensor）并将其类型转换为浮点型（float）

    if prop['task_type'] == 'classification':
        y_train_cls = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)
    return X_train_cls, y_train_cls, X_test, y_test


# 初始化训练所需的模型、优化器和损失函数，并返回
def initialize_training(prop):
    # 使用clsTransformerModel类创建分类模型
    model = cls_transformer_class.clsTransformerModel(prop['task_type'], prop['device'], prop['nclasses'],
                                                      prop['seq_len'], prop['batch'],
                                                      prop['input_size'], prop['emb_size'], prop['nhead'],
                                                      prop['nhid'],prop['nhid_re'], prop['nhid_task'],
                                                      prop['nlayers'], prop['dropout']).to(prop['device'])
    # 将模型移动到prop['device']所指定的设备上

    best_model = cls_transformer_class.clsTransformerModel(prop['task_type'], prop['device'], prop['nclasses'],
                                                           prop['seq_len'], prop['batch'],
                                                           prop['input_size'], prop['emb_size'], prop['nhead'],
                                                           prop['nhid'], prop['nhid_re'], prop['nhid_task'],
                                                           prop['nlayers'], prop['dropout']).to(prop['device'])
    # 用于保存在训练过程中表现最佳的模型

    criterion_re = torch.nn.MSELoss()  # 重建任务的损失函数
    criterion_task = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=prop['lr'])
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=prop['lr'])  # 保存最佳模型的参数

    return model, optimizer, criterion_re, criterion_task, best_model, best_optimizer

# 根据注意力权重和实例权重生成一个掩码的索引
def attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights): # 输入数据，mask比例，high attention比例，实例权重

    res, index = instance_weights.topk(int(math.ceil(ratio_highest_attention * X.shape[1]))) # 使用 topk 函数找到high attention的实例索引，topk返回 值和索引
    index = index.cpu().data.tolist()   # 将实例索引转换为list，并存储在变量 index 中
    index2 = [random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
    # 使用 random.sample 函数从 index 中随机选择 masking_ratio * X.shape[1] 个特征索引，形成一个二维列表 index2
    return np.array(index2)


def random_instance_masking(X, masking_ratio, ratio_highest_attention, instance_weights):   # 输入数据，mask比例，high attention比例，实例权重

    indices = attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights)  # 获取mask后的特征索引，存储在变量 indices 中
    boolean_indices = np.array([[True if i in index else False for i in range(X.shape[1])] for index in indices])
    boolean_indices_masked = np.repeat(boolean_indices[:, :, np.newaxis], X.shape[2], axis=2)
    boolean_indices_unmasked = np.invert(boolean_indices_masked)

    X_train_re, y_train_re_masked, y_train_re_unmasked = np.copy(X), np.copy(X), np.copy(X)
    X_train_re = np.where(boolean_indices_unmasked, X, 0.0)
    y_train_re_masked = y_train_re_masked[boolean_indices_masked].reshape(X.shape[0], -1)
    y_train_re_unmasked = y_train_re_unmasked[boolean_indices_unmasked].reshape(X.shape[0], -1)
    X_train_re, y_train_re_masked, y_train_re_unmasked = torch.as_tensor(X_train_re).float(), torch.as_tensor(
        y_train_re_masked).float(), torch.as_tensor(y_train_re_unmasked).float()

    return X_train_re, y_train_re_masked, y_train_re_unmasked, boolean_indices_masked, boolean_indices_unmasked


def compute_re_loss(model, device, criterion_re, y_train_re_masked, y_train_re_unmasked, batched_input_re, \
                     batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start):
    model.train()
    out_re = model(torch.as_tensor(batched_input_re, device=device), 'reconstruction')[0]

    out_re_masked = torch.as_tensor(
        out_re[torch.as_tensor(batched_boolean_indices_masked)].reshape(out_re.shape[0], -1), device=device)
    out_re_unmasked = torch.as_tensor(
        out_re[torch.as_tensor(batched_boolean_indices_unmasked)].reshape(out_re.shape[0], -1), device=device)

    loss_re_masked = criterion_re(out_re_masked[: num_inst],
                                   torch.as_tensor(y_train_re_masked[start: start + num_inst], device=device))
    loss_re_unmasked = criterion_re(out_re_unmasked[: num_inst],
                                     torch.as_tensor(y_train_re_unmasked[start: start + num_inst], device=device))

    return loss_re_masked, loss_re_unmasked


def compute_task_loss(nclasses, model, device, criterion_task, y_train_cls, batched_input_task, task_type, num_inst,
                      start):
    model.train()
    out_task, attn = model(torch.as_tensor(batched_input_task, device=device), task_type)
    out_task = out_task.view(-1, nclasses) if task_type == 'classification' else out_task.squeeze()
    loss_task = criterion_task(out_task[: num_inst], torch.as_tensor(y_train_cls[start: start + num_inst],
                                                                     device=device))  # dtype = torch.long
    return attn, loss_task


def cls_task_train(model, criterion_re, criterion_task, optimizer, X_train_re, X_train_cls, y_train_re_masked,
                    y_train_re_unmasked, \
                    y_train_cls, boolean_indices_masked, boolean_indices_unmasked, prop):
    model.train()  # Turn on the train mode
    total_loss_re_masked, total_loss_re_unmasked, total_loss_task = 0.0, 0.0, 0.0
    num_batches = math.ceil(X_train_re.shape[0] / prop['batch'])
    output, attn_arr = [], []

    for i in range(num_batches):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        num_inst = y_train_cls[start: end].shape[0]

        optimizer.zero_grad()

        batched_input_re = X_train_re[start: end]
        batched_input_task = X_train_cls[start: end]
        batched_boolean_indices_masked = boolean_indices_masked[start: end]
        batched_boolean_indices_unmasked = boolean_indices_unmasked[start: end]

        loss_re_masked, loss_re_unmasked = compute_re_loss(model, prop['device'], criterion_re, y_train_re_masked,
                                                              y_train_re_unmasked, \
                                                              batched_input_re, batched_boolean_indices_masked,
                                                              batched_boolean_indices_unmasked, num_inst, start)

        attn, loss_task = compute_task_loss(prop['nclasses'], model, prop['device'], criterion_task, y_train_cls, \
                                            batched_input_task, prop['task_type'], num_inst, start)

        total_loss_re_masked += loss_re_masked.item()
        total_loss_re_unmasked += loss_re_unmasked.item()
        total_loss_task += loss_task.item() * num_inst

        # a = list(train_model.parameters())[0].clone()
        loss = prop['task_rate'] * (prop['lamb'] * loss_re_masked + (1 - prop['lamb']) * loss_re_unmasked) + (
                1 - prop['task_rate']) * loss_task
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # b = list(train_model.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))

        # if list(model.parameters())[0].grad is None:
        #    print("None")

        # remove the diagonal values of the attention map while aggregating the column wise attention scores
        attn_arr.append(torch.sum(attn, axis=1) - torch.diagonal(attn, offset=0, dim1=1, dim2=2))

    instance_weights = torch.cat(attn_arr, axis=0)
    return total_loss_re_masked, total_loss_re_unmasked, total_loss_task / y_train_cls.shape[0], instance_weights


def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
    results = []

    if task_type == 'classification':
        loss = criterion(y_pred.view(-1, nclasses), torch.as_tensor(y, device=device)).item()

        pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)
        acc = accuracy_score(target, pred)
        prec = precision_score(target, pred, average=avg)
        rec = recall_score(target, pred, average=avg)
        f1 = f1_score(target, pred, average=avg)

        results.extend([loss, acc, prec, rec, f1])
    else:
        y_pred = y_pred.squeeze()
        y = torch.as_tensor(y, device=device)
        rmse = math.sqrt(((y_pred - y) * (y_pred - y)).sum().data / y_pred.shape[0])
        mae = (torch.abs(y_pred - y).sum().data / y_pred.shape[0]).item()
        results.extend([rmse, mae])
    # per_class_results = precision_recall_fscore_support(target, pred, average = None, labels = list(range(0, nclasses)))

    return results


def test(model, X, y, batch, nclasses, criterion, task_type, device, avg):
    model.eval()  # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)

    output_arr = []
    with torch.no_grad():
        for i in range(num_batches):
            start = int(i * batch)
            end = int((i + 1) * batch)
            num_inst = y[start: end].shape[0]

            out = model(torch.as_tensor(X[start: end], device=device), task_type)[0]
            output_arr.append(out[: num_inst])

    return evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)


def training(model, optimizer, criterion_re, criterion_task, best_model, best_optimizer, X_train_cls, y_train_cls, X_test, y_test, prop):
    re_loss_masked_arr, re_loss_unmasked_arr, re_loss_arr, task_loss_arr, min_task_loss = [], [], [], [], math.inf #存储每个epoch的重构和任务损失的数组
    acc = 0

    instance_weights = torch.as_tensor(torch.rand(X_train_cls.shape[0], prop['seq_len']), device=prop['device'])  # 随机生成，用于计算任务损失时的实例权重

    for epoch in range(1, prop['epochs'] + 1):

        X_train_re, y_train_re_masked, y_train_re_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            random_instance_masking(X_train_cls, prop['masking_ratio'], prop['ratio_highest_attention'], instance_weights)

        re_loss_masked, re_loss_unmasked, task_loss, instance_weights = cls_task_train(model, criterion_re, criterion_task, optimizer,
                                                                                          X_train_re, X_train_cls,
                                                                                          y_train_re_masked, y_train_re_unmasked, y_train_cls,
                                                                                          boolean_indices_masked, boolean_indices_unmasked, prop)

        re_loss_masked_arr.append(re_loss_masked)
        re_loss_unmasked_arr.append(re_loss_unmasked)
        re_loss = re_loss_masked + re_loss_unmasked
        re_loss_arr.append(re_loss)
        task_loss_arr.append(task_loss)
        print('Epoch: ' + str(epoch) + ', Reconstruction Loss: ' + str(re_loss), ', Classification Loss: ' + str(task_loss))

        # save model and optimizer for lowest training loss on the end task
        if task_loss < min_task_loss:
            min_task_loss = task_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer.load_state_dict(optimizer.state_dict())

        # Saved best model state at the lowest training loss is evaluated on the official test set
        test_metrics = test(best_model, X_test, y_test, prop['batch'], prop['nclasses'], criterion_task,
                            prop['task_type'], prop['device'], prop['avg'])

        if prop['task_type'] == 'classification' and test_metrics[1] > acc:
            acc = test_metrics[1]

    if prop['task_type'] == 'classification':
        print('Dataset: ' + prop['dataset'] + ', Acc: ' + str(acc))

    del model
    torch.cuda.empty_cache()
