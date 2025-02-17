# 启动：python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/utils/3dplot.py
import pandas as pd
import plotly.express as px

# 从 CSV 文件加载数据
df = pd.read_csv("/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/enhenced_train.csv")

# 绘制 3D 散点图
fig = px.scatter_3d(
    df,  # 数据源
    x='lp',  # X轴
    y='ln',  # Y轴
    z='hc',  # Z轴
    color='orders',  # 根据 orders 列分组并设置颜色
    size_max=10,
    title='Enhenced-Train-Data',  # 图表标题
    labels={'lp': 'LP', 'ln': 'LN', 'hc': 'HC', 'orders': 'Orders'}  # 图例标签
)

# 更新图例和标记大小
fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

# 更新图例布局
fig.update_layout(
    legend=dict(
        title='Orders',
        yanchor='top',
        y=0.99,
        xanchor='left',
        x=0.01
    )
)

# 显示图表
fig.show()

# 导出为 HTML 文件
fig.write_html("3d_scatter_plot.html")

# 导出为 PNG 图片
fig.write_image("3d_scatter_plot.png")