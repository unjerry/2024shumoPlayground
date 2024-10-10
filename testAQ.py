from aquarel import load_theme
import matplotlib.pyplot as plt
import numpy as np

theme = load_theme("arctic_dark")  # 调用aquarel中的'arctic_dark'
# plt.figure(figsize=(12, 6))

# 创建数据
np.random.seed(10)
data_boxplot = [np.random.normal(0, std, 100) for std in range(1, 5)]
data_lineplot = [np.cumsum(np.random.randn(100)) for _ in range(4)]

# 调用aquarel
theme.apply()
# plt.subplot(121)
# plt.boxplot(data_boxplot)
# plt.xticks([1, 2, 3, 4], ['Group 1', 'Group 2', 'Group 3', 'Group 4'])
# plt.title('arctic_light')

# plt.subplot(122)
plt.rcParams["figure.dpi"] = 300
for i, line_data in enumerate(data_lineplot):
    plt.plot(np.linspace(0, 1000, len(line_data)), line_data, ".-", label=f"curve A{i+1}")
plt.title("solution_curve")

plt.legend()
theme.apply_transforms()
plt.tight_layout()
# plt.show()
plt.savefig("aqtest")
