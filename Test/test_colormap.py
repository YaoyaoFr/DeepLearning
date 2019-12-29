import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# load data
from Analyse.visualize import customize_colormap


def load_data(max: int = 1,
              min: int = -1):  # 文件编号
    data = np.zeros(shape=[10, 10])
    for i in range(10):
        x = np.random.randint(low=0, high=10)
        y = np.random.randint(low=0, high=10)
        data[x, y] = np.random.normal()
        data[y, x] = data[x, y]

    data_min = np.min(data)
    data_max = np.max(data)
    data = data / (data_max - data_min) * (max - min)
    return data


# for i in range(1, 10000):
# 加载数据
data = load_data()
fig = plt.figure()
# 加载图片设置

bwr_map = mpl.cm.bwr
my_cmap = customize_colormap(data)
#
# # 第一个子图,按照默认配置
# ax = fig.add_subplot(221)
# ax.imshow(data)
#
# # 第二个子图,使用api自带的colormap
# ax = fig.add_subplot(222)
# cmap = mpl.cm.bwr  # 蓝，白，红
# ax.imshow(data, cmap=cmap)

# 第三个子图增加一个colorbar
# ax = fig.add_subplot(223)
ax = fig.add_subplot(121)
im = plt.imshow(data, cmap=my_cmap)
plt.colorbar(im)  # 增加colorbar

# # 第四个子图可以调整colorbar
# ax = fig.add_subplot(224)
# cmap = mpl.cm.rainbow
# # 这里设置colormap的固定值
# norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# im = ax.imshow(data, cmap=cmap)
# plt.colorbar(im, cmap=cmap, norm=norm, ticks=[-1, 0, 1])

# 显示
plt.show()
