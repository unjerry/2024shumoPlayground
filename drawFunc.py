import torch
import matplotlib.pyplot as plt
from aquarel import load_theme

sp = torch.nn.Softplus(beta=0.1)
X = torch.linspace(0, 1000, 100)
theme = load_theme("arctic_light")  # 调用aquarel中的'arctic_dark'
theme.apply()
plt.rcParams["figure.dpi"] = 500
plt.title("g")
plt.gca().set_aspect(1.0)
Y = sp(X - 500)
plt.plot(X, Y, ".-")
theme.apply_transforms()
plt.savefig("func_fig_3")
plt.clf()

plt.rcParams["figure.dpi"] = 500
plt.title("g")
plt.gca().set_aspect(1.0)
Y = sp(500 - X)
plt.plot(X, Y, ".-")
theme.apply_transforms()
plt.savefig("func_fig_4")
plt.clf()
