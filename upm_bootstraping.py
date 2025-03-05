import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体支持中文
# rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['font.family'] = ['Arial', 'SimHei']  # 按优先级列出多个字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号


def estimate_statistics_and_outliers(samples, th_min, th_max):
    """
    估计总体均值和方差，并计算每100万样本中超出阈值的样本数量。

    参数：
        samples (list or numpy.ndarray): 输入的N个样本。
        th_min (float): 最小阈值。
        th_max (float): 最大阈值。

    返回：
        tuple: (总体均值, 总体方差, 每100万样本中超出阈值的样本数量)
    """
    # 转换为numpy数组
    samples = np.array(samples)

    # 计算均值和方差
    mean = np.mean(samples)
    variance = np.var(samples, ddof=1)  # 使用ddof=1计算样本方差
    std_dev = np.sqrt(variance)

    # 根据高斯分布计算超出阈值的样本比例
    below_min_prob = norm.cdf(th_min, loc=mean, scale=std_dev)
    above_max_prob = 1 - norm.cdf(th_max, loc=mean, scale=std_dev)
    outlier_ratio = below_min_prob + above_max_prob

    # 估计每100万样本中不合格样本的数量
    outliers_per_million = int(outlier_ratio * 1_000_000)

    return mean, variance, outliers_per_million


def bootstrap_outlier_estimation(samples, th_min, th_max, num_bootstrap=1000):
    """
    使用Bootstraping估计每100万样本中不合格样本数量的分布。

    参数：
        samples (list or numpy.ndarray): 输入的N个样本。
        th_min (float): 最小阈值。
        th_max (float): 最大阈值。
        num_bootstrap (int): Bootstrap重采样次数。

    返回：
        tuple: (每100万样本中不合格样本数量的均值, 95%置信区间)
    """
    samples = np.array(samples)
    n = len(samples)
    bootstrap_outliers = []

    for _ in range(num_bootstrap):
        # 重采样
        bootstrap_sample = np.random.choice(samples, size=n, replace=True)
        mean = np.mean(bootstrap_sample)
        std_dev = np.std(bootstrap_sample, ddof=1)

        # 根据高斯分布计算超出阈值的样本比例
        below_min_prob = norm.cdf(th_min, loc=mean, scale=std_dev)
        above_max_prob = 1 - norm.cdf(th_max, loc=mean, scale=std_dev)
        outlier_ratio = below_min_prob + above_max_prob

        # 记录每100万样本中的不合格数量
        bootstrap_outliers.append(outlier_ratio * 1_000_000)

    # 计算均值和95%置信区间
    mean_outliers = np.mean(bootstrap_outliers)
    ci_lower = np.percentile(bootstrap_outliers, 2.5)
    ci_upper = np.percentile(bootstrap_outliers, 97.5)

    return mean_outliers, (ci_lower, ci_upper), bootstrap_outliers


def plot_histogram_with_gaussian(samples, mean, std_dev, th_min, th_max):
    """
    绘制样本直方图和估计的高斯分布。

    参数：
        samples (list or numpy.ndarray): 输入的样本数据。
        mean (float): 样本均值。
        std_dev (float): 样本标准差。
        th_min (float): 最小阈值。
        th_max (float): 最大阈值。
    """
    # 确定绘图范围
    x_min = max(mean - 8 * std_dev, th_min)
    x_max = min(mean + 8 * std_dev, th_max)

    # 绘制直方图
    plt.figure()
    plt.hist(samples, bins=20, density=True, alpha=0.6, color='g', label='样本直方图')

    # 绘制高斯分布曲线
    x = np.linspace(x_min, x_max, 1000)
    y = norm.pdf(x, loc=mean, scale=std_dev)
    plt.plot(x, y, 'r-', label='估计的高斯分布')

    # 添加标签和图例
    plt.title('样本直方图与估计的高斯分布')
    plt.xlabel('样本值')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid()


def plot_bootstrap_distribution(bootstrap_outliers, ci_mean, ci_lower, ci_upper):
    """
    绘制Bootstrap估计的不合格样本数量分布及置信区间。

    参数：
        bootstrap_outliers (list or numpy.ndarray): Bootstrap估计的每100万样本不合格数量。
        ci_lower (float): 置信区间下界。
        ci_upper (float): 置信区间上界。
    """
    bins = np.logspace(np.log10(min(bootstrap_outliers) + 1e-4), np.log10(max(bootstrap_outliers) + 1e-4), num=50) if max(bootstrap_outliers) > 1 else 10
    # 绘制直方图
    plt.figure()
    plt.hist(bootstrap_outliers, bins=bins, density=False, alpha=0.6, color='r', label='UPM分布')
    plt.xscale('log')  # 设置x轴为对数尺度
    plt.title('Bootstraping-UPM分布直方图')
    plt.xlabel('每100万样本不合格数量')
    plt.ylabel('直方图计数')
    # 添加置信区间标记
    plt.axvline(ci_mean, color='r', linestyle='--', label='UPM均值')
    plt.axvline(ci_lower, color='g', linestyle='--', label='95%置信区间下界')
    plt.axvline(ci_upper, color='b', linestyle='--', label='95%置信区间上界')
    plt.legend()
    plt.grid()

    # 绘制直方图
    plt.figure()
    plt.hist(bootstrap_outliers, bins=bins, density=True, alpha=0.6, color='r', label='UPM分布')
    plt.title('Bootstraping-UPM分布直方图(x轴对数)')
    plt.xlabel('每100万样本不合格数量')
    plt.ylabel('概率密度')
    # 添加置信区间标记
    plt.axvline(ci_mean, color='r', linestyle='--', label='UPM均值')
    plt.axvline(ci_lower, color='g', linestyle='--', label='95%置信区间下界')
    plt.axvline(ci_upper, color='b', linestyle='--', label='95%置信区间上界')
    plt.legend()
    plt.grid()


# 示例输入
if __name__ == "__main__":
    # 样本数据
    np.random.seed(0)  # 生成样本时固定随机种子
    N = 100  # 样本数
    samples = np.random.normal(loc=20, scale=1.0, size=N)  # 均值为20，方差为1的正态分布样本

    # 阈值, 超出阈值范围视为不合格
    th_min = 16
    th_max = 24

    # Bootstraping设置
    np.random.seed(None)  # Bootstraping时不固定随机种子
    num_bootstrap = 5000  # Bootstraping抽样次数

    # 直接高斯分布估计
    mean, variance, outliers = estimate_statistics_and_outliers(samples, th_min, th_max)
    # 输出结果
    print(f"总体均值: {mean:.4f}")
    print(f"总体方差: {variance:.4f}")
    print(f"直接高斯分布估计的每100万样本中不合格的样本数量: {outliers}")
    # 直接高斯分布估计绘图
    plot_histogram_with_gaussian(samples, mean, np.sqrt(variance), th_min, th_max)

    # 使用Bootstraping估计
    mean_outliers, ci_outliers, bootstrap_outliers = bootstrap_outlier_estimation(samples, th_min, th_max, num_bootstrap=num_bootstrap)
    # 输出结果
    print(f"Bootstraping估计的每100万样本中不合格的数量均值: {mean_outliers:.2f}")
    print(f"Bootstraping估计的不合格的数量的95%置信区间: ({ci_outliers[0]:.2f}, {ci_outliers[1]:.2f})")
    p = (np.array(bootstrap_outliers) > 50).sum() / len(bootstrap_outliers)
    print(f"Bootstraping估计的UPM>50的概率: {p * 100:.1f}%")
    p = (np.array(bootstrap_outliers) < 1).sum() / len(bootstrap_outliers)
    print(f"Bootstraping估计的UPM<1的概率: {p * 100:.1f}%")
    # Bootstraping绘图
    plot_bootstrap_distribution(bootstrap_outliers, mean_outliers, ci_outliers[0], ci_outliers[1])

    plt.show()
