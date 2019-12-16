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
    """  计算给定数据集的香农熵)
    Args:
        dataSet 数据集
    Returns:
        该数据集所有类别标签包含的信息熵
    """
    # 统计标签出现的次数
    label_count = Counter(data[-1] for data in dataSet)
    # 计算概率
    probs = [p[1]*1.0/ len(dataSet) for p in label_count.items()] #p[1]是某个类别出现的次数,p[0]是该类别的名称
    # 计算香农熵
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt

def splitDataSet(dataSet, index, value):
    """
    将指定特征列的特征值等于 value 的行除index以外剩下的列作为子数据集。

    """
    retDataSet = [data[:index] + data[index + 1:] for data in dataSet for i, v in enumerate(data) if i == index and v == value]
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征)
    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    """
    # 求第一行有多少列的 Feature, 最后一列是label列嘛
    numFeatures = len(dataSet[0]) - 1
    # 数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1
    # iterate over all the features
    for i in range(numFeatures):
        # 获取对应的feature下的所有数据
        featList = [example[i] for example in dataSet]
        # 获取剔重后的集合，使用set对list数据进行去重
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        print('第{0} 列的信息增益是:{1:.3f}'.format(i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    print('最好的特征列是:',bestFeature)
    return bestFeature

def majorityCnt(classList):
    """
    majorityCnt(选择出现次数最多的一个结果)
    用于判断当数据集已经处理了所有属性,但类标签依然不唯一这种情况
    """
    major_label = Counter(classList).most_common(1)[0]
    return major_label

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的列，得到最优列对应的label的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]  
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
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
    # 获取tree的根节点对应的key值,书上这里会报错,必须加list
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型,如果是则继续递归向下寻找
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

    # print("加后",labels)

def grabTree(filename):
    #将之前存储的决策树模型使用pickle模块还原出来
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def fishTest():
    import copy
    #1.创建数据和结果标签
    myDat,labels = createDataSet()
    myTree = createTree(myDat, copy.deepcopy(labels))
    print("决策树是:",myTree)
    #2.分类树的存取
    storeTree(myTree,'dt.txt')
    #3.测试决策树分类器,测试数据为(1,1)
    print("分类的结果是:",classify(myTree, labels, [1, 1]))
    #4.获得树的高度
    print("树的高度是:",get_tree_height(myTree))
    #5.画图可视化展现fishTest
    dtPlot.createPlot(myTree)
    #6.打印取出的树
    print("存储后取出的树是:",grabTree('dt.txt'))

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

# ------项目案例2: 预测隐形眼镜类型---------

def ContactLensesTest():
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('data/lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print("决策树是:",lensesTree)
    print("树的高度是:",get_tree_height(lensesTree))
    # 画图可视化展现
    dtPlot.createPlot(lensesTree)

if __name__ == "__main__":
  
    fishTest()
    print("="*40)
    ContactLensesTest()
