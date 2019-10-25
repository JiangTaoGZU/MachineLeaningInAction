from math import log
import decisionTreePlot as dtPlot
from collections import Counter

# ------项目案例1: 海洋生物预测---------

def createDataSet():
    dataSet = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    """calcShannonEnt(calculate Shannon entropy 计算给定数据集的香农熵)
    Args:
        dataSet 数据集
    Returns:
        返回 每一组feature下的某个分类下，香农熵的信息期望
    """
    # 统计标签出现的次数
    label_count = Counter(data[-1] for data in dataSet)
    # 计算概率
    probs = [ p[1] / len(dataSet)for p in label_count.items()]
    # 计算香农熵
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征)
    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列的索引
    """
    # 计算初始香农熵
    base_entropy = calcShannonEnt(dataSet)
    best_info_gain = 0
    best_feature = -1
    # 遍历每一个特征
    for i in range(len(dataSet[0]) - 1):
        # 对当前特征进行统计
        feature_count = Counter([data[i] for data in dataSet])
        # 计算各列的各value值被分割数据集后的香农熵
        new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
                       for feature in feature_count.items())
        # 更新值
        info_gain = base_entropy - new_entropy
        print('第{0} 列的信息增益是:{1:.3f}'.format(i, info_gain))
        if  info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    print("最好的特征列是:",best_feature)
    return best_feature

def majorityCnt(classList):
    """majorityCnt(选择出现次数最多的一个结果)
        Args:
            classList label列的集合
        Returns:
            bestFeature 最优的特征列
        """
    major_label = Counter(classList).most_common(1)[0]
    return major_label

def createTree(dataSet, labels):
    """
    Desc:
        创建决策树
    Args:
        dataSet -- 要创建决策树的训练数据集
        labels -- 训练数据集中特征对应的含义的labels，不是目标变量
    Returns:
        myTree -- 创建完成的决策树
    """
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的列，得到最优列对应的label的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    labels.append(bestFeatLabel)
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list]
    del(labels[bestFeat])
    # 取出最优列的所有value值，然后对它的分支做递归建树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree

def classify(inputTree, featLabels, testVec):
    """
    Desc:
        对新数据进行分类
    Args:
        inputTree  -- 已经训练好的决策树模型
        featLabels -- Feature标签对应的名称，不是目标变量
        testVec    -- 测试输入的数据
    Returns:
        classLabel -- 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值,书上这里会报错,必须加list
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    """
    Desc:
        将之前训练好的决策树模型存储起来，使用 pickle 模块
    Args:
        inputTree -- 以前训练好的决策树模型
        filename -- 要存储的名称
    Returns:
        None
    """
    import pickle

    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    #将之前存储的决策树模型使用pickle模块还原出来
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def fishTest():
    import copy
    # 1.创建数据和结果标签
    myDat,labels = createDataSet()
    myTree = createTree(myDat, copy.deepcopy(labels))
    print("决策树是:",myTree)
    #2.分类树的存取
    storeTree(myTree,'dt.txt')
    #3.[1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))
    #4.获得树的高度
    print("树的高度是:",get_tree_height(myTree))
    #5.画图可视化展现fishTest
    dtPlot.createPlot(myTree)

def get_tree_height(tree):
    """
     Desc:
        递归获得决策树的高度
    Args:
        tree
    Returns:
        树高
    """

    if not isinstance(tree, dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1

# ------项目案例1: 预测隐形眼镜类型---------

def ContactLensesTest():
    """
    Desc:
        预测隐形眼镜的测试代码
    Args:
        none
    Returns:
        none
    """
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('data/lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    dtPlot.createPlot(lensesTree)

if __name__ == "__main__":
    print("取出的树是:",grabTree('dt.txt'))
    fishTest()
    ContactLensesTest()
