import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from aquarel import load_theme

rt = 10.0

th = torch.linspace(0, torch.pi * 2, 100)
uni = torch.tensor([5.0 * rt, 0.0])
matt = torch.stack(
    [
        torch.stack([torch.cos(th), -torch.sin(th)], dim=1),
        torch.stack([torch.sin(th), torch.cos(th)], dim=1),
    ],
    dim=1,
)
Pp = torch.einsum("nij,j->ni", matt, uni)


class M(nn.Module):
    def __init__(self, n, p0, p1, dl, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vec = nn.Parameter(torch.zeros([n], requires_grad=True))
        self.vec2 = nn.Parameter(torch.zeros([n], requires_grad=True))
        self.sig = nn.Sigmoid()
        self.P0 = nn.Parameter(p0, requires_grad=False)
        self.P1 = nn.Parameter(p1, requires_grad=False)
        self.dl = nn.Parameter(torch.tensor(dl), requires_grad=False)
        self.theh = nn.Parameter(torch.tensor([torch.pi / 6]), requires_grad=True)
        self.theh2 = nn.Parameter(
            torch.tensor([torch.pi + torch.arcsin(torch.ones([]) / 7.0)]),
            requires_grad=True,
        )
        self.unit = nn.Parameter(torch.tensor([1.0, 0]), requires_grad=False)

    def forward(self):
        thetas = (2 * self.sig(self.vec) - 1) * torch.pi / 3
        theta2 = (2 * self.sig(self.vec2) - 1) * torch.pi / 3
        # print("pre", pre)
        mat0 = torch.stack(
            [
                torch.concatenate([torch.cos(self.theh), -torch.sin(self.theh)], dim=0),
                torch.concatenate([torch.sin(self.theh), torch.cos(self.theh)], dim=0),
            ],
            dim=0,
        )
        mat1 = torch.stack(
            [
                torch.concatenate(
                    [torch.cos(self.theh2), -torch.sin(self.theh2)], dim=0
                ),
                torch.concatenate(
                    [torch.sin(self.theh2), torch.cos(self.theh2)], dim=0
                ),
            ],
            dim=0,
        )
        # print(mat0.shape)
        unit = torch.einsum("ij,j->i", mat0, self.unit)
        unit2 = torch.einsum("ij,j->i", mat1, self.unit)
        pre = self.dl * unit
        pre2 = self.dl * unit2
        matt1 = torch.stack(
            [
                torch.stack([torch.cos(thetas), -torch.sin(thetas)], dim=1),
                torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=1),
            ],
            dim=1,
        )
        matt2 = torch.stack(
            [
                torch.stack([torch.cos(theta2), -torch.sin(theta2)], dim=1),
                torch.stack([torch.sin(theta2), torch.cos(theta2)], dim=1),
            ],
            dim=1,
        )
        # print(mat)
        P = self.P0
        P2 = self.P1
        for i in range(thetas.shape[0]):
            # print(thetas[i], mat[i])
            pre = matt1[i] @ pre
            P = torch.concatenate((P, (P[-1, :] + pre).view(1, -1)), dim=0)
            pre2 = matt2[i] @ pre2
            P2 = torch.concatenate((P2, (P2[-1, :] + pre2).view(1, -1)), dim=0)
        # print(P)
        return P, P2


p0 = torch.tensor([[-10.0 * rt, 0.0]], device="cpu")
p1 = torch.tensor([[35.0 * rt, 0.0]], device="cpu")
len = 45 * rt
n = 150
model = M(n, p0, p1, 0.10 * rt)
lable = "1_2"
model = torch.load("model" + lable)
model.to("cpu")
print(model.vec[:].shape[0])
z = model()
print(model.P0[0])
print(model.P1[0])
print(model.dl)
print(3)
print(z[0][-1])
print(z[1][-1])
print(torch.dist(z[0][-1], model.P1[0]))
print(torch.dist(z[1][-1], model.P0[0]))
print(torch.dist(z[0][-1], model.P1[0]) + model.vec[:].shape[0] * model.dl)
print(torch.dist(z[1][-1], model.P0[0]) + model.vec[:].shape[0] * 3)
print((torch.dist(z[0][-1], model.P1[0]) + model.vec[:].shape[0] * model.dl) / model.dl)
print(
    (torch.dist(z[1][-1], model.P0[0]) + model.vec[:].shape[0] * 3) / 3
)
with open("result" + lable, "w") as file:
    file.write(
        f"A_total_distance:{torch.dist(z[0][-1], model.P1[0]) + model.vec[:].shape[0] * model.dl}\n"
    )
    file.write(
        f"B_total_distance:{torch.dist(z[1][-1], model.P0[0]) + model.vec[:].shape[0] *3}\n"
    )
    file.write(
        f"A_total_time:{(torch.dist(z[0][-1], model.P1[0]) + model.vec[:].shape[0] * model.dl) / model.dl}\n"
    )
    file.write(
        f"B_total_time:{(torch.dist(z[1][-1], model.P0[0]) + model.vec[:].shape[0] * 3) / 3}\n"
    )
theme = load_theme("arctic_light")  # 调用aquarel中的'arctic_dark'
theme.apply()
plt.rcParams["figure.dpi"] = 500
plt.title("solution_curve")
plt.gca().set_aspect(1.0)
# plt.xlim((-20 * rt, 40 * rt))
plt.ylim((-15 * rt, 15 * rt))
plt.plot(Pp[:, 0], Pp[:, 1], "-", label="Circle")
plt.plot(
    z[0].detach().cpu()[:, 0],
    z[0].detach().cpu()[:, 1],
    ".-",
    torch.linspace(z[0][-1][0], model.P1[0][0], 100),
    torch.linspace(z[0][-1][1], model.P1[0][1], 100),
    ".-",
    label="Curve A",
)
# print(z[1])
plt.plot(
    z[1].detach().cpu()[:, 0],
    z[1].detach().cpu()[:, 1],
    ".-",
    torch.linspace(z[1][-1][0], model.P0[0][0], 100),
    torch.linspace(z[1][-1][1], model.P0[0][1], 100),
    ".-",
    label="Curve B",
)
plt.legend()
theme.apply_transforms()
# plt.tight_layout()
# plt.show()
plt.savefig("newfig" + lable)
plt.clf()
# losss = torch.load("Los5-3000-30")
# Los = []
# for it in losss:
#     Los.append(it.item())
# plt.plot(Los, "*-", label="loss")
# plt.legend()
# plt.title("decreasing_curve")
# theme.apply_transforms()
# plt.savefig("lossPic5-3000-30")
