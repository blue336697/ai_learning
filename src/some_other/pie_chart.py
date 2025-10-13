import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
labels = ['电商类网站', '社交媒体类网站', '金融类网站', '票务类网站', '新闻资讯类网站']
sizes = [30, 25, 20, 15, 10]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
explode = (0.1, 0, 0, 0, 0)  # 突出显示第一个扇形

# 生成饼图
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

# 设置饼图的纵横比为正圆形
plt.axis('equal')

# 显示图表
plt.title('爬取源分布')
plt.show()