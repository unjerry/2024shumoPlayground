import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())

rt = 10

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
plt.scatter(Pp[:, 0], Pp[:, 1])
# plt.show()


class M(nn.Module):
    def __init__(self, n, p0, p1, dl, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vec = nn.Parameter(torch.zeros([n], requires_grad=True))
        self.vec2 = nn.Parameter(torch.zeros([n], requires_grad=True))
        self.sig = nn.Sigmoid()
        self.P0 = nn.Parameter(p0, requires_grad=False)
        self.P1 = nn.Parameter(p1, requires_grad=False)
        self.dl = nn.Parameter(torch.tensor(dl), requires_grad=False)
        self.theh = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.theh2 = nn.Parameter(torch.tensor([-2.0]), requires_grad=True)
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


def cost_function(x, y):
    end = torch.tensor([35.0 * rt, 0.0], device="cpu")
    end2 = torch.tensor([-10.0 * rt, 0.0], device="cpu")
    circ = torch.tensor([0.0, 0.0], device="cpu")
    dif = x[-1] - end
    dif2 = y[-1] - end2
    dist = torch.einsum("i,i->", dif, dif)
    dist2 = torch.einsum("i,i->", dif2, dif2)
    # print(dist)
    dd = torch.einsum("i,i->", dif, dif)
    dd2 = torch.einsum("i,i->", dif2, dif2)
    # / (
    #     torch.sqrt(
    #         torch.einsum("i,i->", torch.tensor([5500, 0]), torch.tensor([5500, 0]))
    #     )
    # )
    ans = dd + dd2
    # ans = 0
    for p in x:
        # print(p)
        # print(
        #     torch.exp(
        #         -5 * (torch.einsum("i,i->", (p - circ) / 500, (p - circ) / 500))
        #     ).tolist()
        # )
        ans += dist * torch.log(1.0 + torch.exp((5.0 * rt - torch.dist(p, circ))))
    for p in y:
        # print(p)
        # print(
        #     torch.exp(
        #         -5 * (torch.einsum("i,i->", (p - circ) / 500, (p - circ) / 500))
        #     ).tolist()
        # )
        ans += dist2 * torch.log(1.0 + torch.exp((5.0 * rt - torch.dist(p, circ))))

    return ans


p0 = torch.tensor([[-10.0 * rt, 0.0]], device="cpu")
p1 = torch.tensor([[35.0 * rt, 0.0]], device="cpu")
len = 45 * rt
n = 66
model = M(n, p0, p1, 0.3 * rt)
model.to("cpu")
opt = torch.optim.Adam(model.parameters())

# z = model().detach().cpu()
# print(z)
# plt.scatter(z[:, 0], z[:, 1])
# plt.gca().set_aspect(1.0)
# plt.savefig("figs/tmp1.png")

preloss = torch.tensor(0.0)
for i in range(100000):
    opt.zero_grad()
    z = model()
    loss = cost_function(z[0], z[1])
    loss.backward()
    opt.step()
    torch.set_printoptions(precision=4, linewidth=1000)
    print(
        f"{i}\t{z[0][-1]}\t{loss.tolist():.4f}\t{preloss.tolist():.4f}\t{(torch.abs(preloss-loss)).tolist():.4f}"
    )
    if torch.abs(preloss - loss).tolist() < 1e-2:
        break
    preloss = loss
    if (i + 1) % 1000 == 0:
        z = model()
        plt.gca().set_aspect(1.0)
        plt.xlim((-15 * rt, 40 * rt))
        plt.ylim((-55 * rt / 2, 55 * rt / 2))
        plt.scatter(z[0].detach().cpu()[:, 0], z[0].detach().cpu()[:, 1])
        # print(z[1])
        plt.scatter(z[1].detach().cpu()[:, 0], z[1].detach().cpu()[:, 1])
        plt.savefig(f"figs/tmp_{i}.png")
        # plt.show()

# z = model().detach().cpu()
# print(z, z.device, sep="\n")

# import matplotlib.pyplot as plt

# plt.scatter(z[:, 0], z[:, 1], alpha=1)
# plt.gca().set_aspect(1.0)
# plt.xlim((-15 * rt, 40 * rt))
# plt.ylim((-55 * rt / 2, 55 * rt / 2))
# plt.savefig("figs/tmp.png")
# plt.show()
