from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt

def loadDataSet(fileName):
    '''
    加载数据集
    :param fileName:
    :return:
    '''
    # 初始化一个空列表
    dataSet = []
    # 读取文件
    fr = open(fileName)
    # 循环遍历文件所有行
    for line in fr.readlines():
        # 切割每一行的数据
        curLine = line.strip().split('\t')
        # 将数据转换为浮点类型,便于后面的计算
        # 将数据追加到dataMat
        fltLine = list(map(float,curLine))    # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)
    # 返回dataMat
    return dataSet


def distEclud(vecA, vecB):
    '''
    欧氏距离计算函数
    :param vecA:
    :param vecB:
    :return:
    '''
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataMat, k):
    '''
    为给定数据集构建一个包含K个随机质心的集合,
    随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成
    然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
    :param dataMat:
    :param k:
    :return:
    '''
    # 获取样本数与特征值
    m, n = shape(dataMat)
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = mat(zeros((k, n)))
    # 循环遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataMat[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # 计算每一列的质心,并将值赋给centroids
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    # 返回质心
    return centroids


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    '''
    创建K个质心,然后将每个点分配到最近的质心,再重新计算质心。
    这个过程重复数次,直到数据点的簇分配结果不再改变为止
    :param dataMat: 数据集
    :param k: 簇的数目
    :param distMeans: 计算距离
    :param createCent: 创建初始质心
    :return:
    '''
    # 获取样本数和特征数
    m, n = shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = mat(zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment


def biKmeans(dataMat, k, distMeas=distEclud):
    '''
    在给定数据集,所期望的簇数目和距离计算方法的条件下,函数返回聚类结果
    :param dataMat:
    :param k:
    :param distMeas:
    :return:
    '''
    m, n = shape(dataMat)
    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m, 2)))
    # 计算整个数据集的质心,并使用一个列表来保留所有的质心
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList = [centroid0]
    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataMat[j, :]) ** 2
        # print("ssssasa\n",clusterAssment)
    # 对簇不停的进行划分,直到得到想要的簇数目为止
    while (len(centList) < k):
        # 初始化最小SSE为无穷大,用于比较划分前后的SSE
        lowestSSE = inf
        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(centList)):
            # 对每一个簇,将该簇中的所有点存成一个小的数据集
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
            # kMeans会生成两个质心(簇),同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 将误差值与剩余数据集的误差之和作为本次划分的误差
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit: ",sseSplit,sseNotSplit) 
            # 如果本次划分的SSE值最小,则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果
        # 调用kmeans函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇,
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 更新为最佳质心
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('最优的划分簇是: ', bestCentToSplit)
        print('bestClustAss的长度= ', len(bestClustAss))
        # 更新质心列表
        # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加bestNewCents的第二个质心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配最好簇下的数据(质心)以及SSE
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


#---------------------简单测试-----------------------------
def testBasicFunc():
    # 加载测试数据集
    dataMat = mat(loadDataSet('data/10.KMeans/testSet.txt'))

    # 测试 randCent() 函数是否正常运行。
    # 首先，先看一下矩阵中的最大值与最小值
    print('min(dataMat[:, 0])=', min(dataMat[:, 0]))
    print('min(dataMat[:, 1])=', min(dataMat[:, 1]))
    print('max(dataMat[:, 1])=', max(dataMat[:, 1]))
    print('max(dataMat[:, 0])=', max(dataMat[:, 0]))

    # 然后看看 randCent() 函数能否生成 min 到 max 之间的值
    print('randCent(dataMat, 2)=', randCent(dataMat, 2))

    # 最后测试一下距离计算方法
    print(' distEclud(dataMat[0], dataMat[1])=', distEclud(dataMat[0], dataMat[1]))

#---------------------测试Kmeans--------------------------
def testKMeans():
    # 加载测试数据集
    dataMat = mat(loadDataSet('data/10.KMeans/testSet.txt'))
    # 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
    # 这个过程重复数次，知道数据点的簇分配结果不再改变位置。
    # 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似）
    myCentroids, clustAssing = kMeans(dataMat, 4)
    print('centroids=\n', myCentroids)
    print('clusterAssment\n', clustAssing)

#---------------------测试二分KMeans--------------------------
def testBiKMeans():
    # 加载测试数据集
    dataMat = mat(loadDataSet('data/10.KMeans/testSet2.txt'))
    centList, myNewAssments = biKmeans(dataMat, 3)
    print('centList=\n', centList)

if __name__ == "__main__":

    # 测试基础的函数
    testBasicFunc()
    print("="*40)
    # 测试 kMeans 函数
    testKMeans()
    print("="*40)
    # 测试二分 biKMeans 函数
    testBiKMeans()