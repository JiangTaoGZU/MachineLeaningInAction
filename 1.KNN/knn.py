from numpy import *
from os import listdir
import matplotlib.pyplot as plt

# ------项目案例1: 预测电影类型---------

def createDataSet():
    """
    Desc:
        创建数据集和标签
    Args:
        None
    Returns:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    Desc:
        kNN 的分类函数
    Args:
        inX -- 用于分类的输入向量/测试数据
        dataSet -- 训练数据集的 features
        labels -- 训练数据集的 labels
        k -- 选择最近邻的数目
    Returns:
        sortedClassCount[0][0] -- 输入向量的预测分类 labels
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    """
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet      # tile生成和训练样本对应的矩阵，并与训练样本求差,1带表重复次数
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)      # 将矩阵的每一行相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  #  argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount

def test1():
    """
    第一个例子演示
    """
    group, labels = createDataSet()
    print("电影数据集为:\n",str(group))
    print("电影标签为:",str(labels))
    print("预测电影类型为:",classify0([0.1, 0.1], group, labels, 3))

def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵returnMat和对应的类别classLabelVector
    """
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]   #：左边表示对矩阵行的操作，右边表示对列的操作
        # 添加每列的类别数据到最后，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # print(returnMat)
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals) / ranges
    return normDataSet, ranges, minVals

# ------项目案例2: 约会网站匹配---------

def datingClassTest():
    #设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    #从文件中加载数据
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')

    #分析数据，画散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('约会案例的测试样本数量=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("约会案例分类器的结果是: %d, 正确答案是: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("约会案例错误率是: %f" % (errorCount / float(numTestVecs)))
    print('约会案例错误数量:',errorCount,)

# ------项目案例3: 手写数字识别---------
def img2vector(filename):
    """
    将图像数据转换为向量
    :param filename: 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    :return: 一维矩阵
    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    # 1. 导入数据
    hwLabels = []
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('data/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类识别数是: %d, 实际数字是: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("手写错误的个数是: %d" % errorCount)
    print("手写案例错误率是: %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    test1()
    print("="*40)
    datingClassTest()
    print("="*40)
    handwritingClassTest()