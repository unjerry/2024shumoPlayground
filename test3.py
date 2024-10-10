import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())

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
        self.dl2 = nn.Parameter(torch.tensor(dl), requires_grad=False)
        self.theh = nn.Parameter(torch.tensor([torch.pi / 6]), requires_grad=True)
        self.theh2 = nn.Parameter(
            torch.tensor([torch.pi + torch.arcsin(torch.ones([]) / 7.0)]),
            requires_grad=True,
        )
        self.unit = nn.Parameter(torch.tensor([1.0, 0]), requires_grad=False)

    def forward(self):
        thetas = (2 * self.sig(self.vec) - 1) * 2 * torch.arcsin(self.dl / 0.6 * rt)
        theta2 = (2 * self.sig(self.vec2) - 1) * 2 * torch.arcsin(self.dl2 / 0.6 * rt)
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
        pre2 = self.dl2 * unit2
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
    softplus = torch.nn.Softplus(beta=1000.0 / rt)
    end = torch.tensor([35.0 * rt, 0.0], device="cpu")
    end2 = torch.tensor([-10.0 * rt, 0.0], device="cpu")
    circ = torch.tensor([0.0, 0.0], device="cpu")
    dif = x[-1] - end
    dif2 = y[-1] - end2
    dist = torch.einsum("i,i->", dif, dif)
    # dist = 0
    dist2 = torch.einsum("i,i->", dif2, dif2)
    # dist2 = 0
    # print(dist)
    dd = torch.einsum("i,i->", dif, dif)
    # dd2 = 0
    dd2 = 0.01 * torch.einsum("i,i->", dif2, dif2)
    # / (
    #     torch.sqrt(
    #         torch.einsum("i,i->", torch.tensor([5500, 0]), torch.tensor([5500, 0]))
    #     )
    # )
    ans = dd + dd2
    # ans = 0
    for p in x:
        # print(p)
        # # print(
        # #     torch.exp(
        # #         -5 * (torch.einsum("i,i->", (p - circ) / 500, (p - circ) / 500))
        # #     ).tolist()
        # # )
        ans += (dist2 + dist) * softplus(
            1.0 + torch.exp((5.0 * rt - torch.dist(p, circ)))
        )

    for p in y:
        # print(p)
        # # print(
        # #     torch.exp(
        # #         -5 * (torch.einsum("i,i->", (p - circ) / 500, (p - circ) / 500))
        # #     ).tolist()
        # # )
        ans += (dist2 + dist) * softplus(
            1.0 + torch.exp((5.0 * rt - torch.dist(p, circ)))
        )

    for i in range(x.shape[0]):
        # print(i)
        meet = y[i] - x[i]
        # print("x: ", meet, " end")
        norm = torch.torch.tensor([-1 * meet[1], meet[0]], device="cpu")
        D = torch.einsum("i,i->", norm, x[i]) / torch.dist(norm, circ)
        # print(torch.dist(norm, circ))
        # print(dist2.shape, torch.dist(norm, circ)-5.0*rt)
        # print(dist, dist2, dist + dist2)
        # ans += (dist + dist2) * (
        #     softplus((D - 5.0 * rt))
        #     # - softplus(-5.0 * rt * torch.ones([]))
        #     # - torch.exp(-5.0 * rt * torch.ones([])) * D
        # )
        ans += (dist + dist2) * (softplus((torch.abs(D) - 5.0 * rt)))
        ans += (dist + dist2) * softplus(
            torch.einsum("i,i->", meet, x[i]) / torch.dist(meet, circ)
        )
        ans += (dist + dist2) * softplus(
            torch.einsum("i,i->", meet, -y[i]) / torch.dist(meet, circ)
        )

    return ans


p0 = torch.tensor([[-10.0 * rt, 0.0]], device="cpu")
p1 = torch.tensor([[35.0 * rt, 0.0]], device="cpu")
len = 45 * rt
n = 135
model = M(n, p0, p1, 0.3 * rt)
# model = torch.load("model")
model.to("cpu")
# model.P0 = p0
opt = torch.optim.Adam(model.parameters())

# z = model().detach().cpu()
# print(z)
# plt.scatter(z[:, 0], z[:, 1])
# plt.gca().set_aspect(1.0)
# plt.savefig("figs/tmp1.png")

preloss = torch.tensor(0.0)
for k in range(-10, -11, -1):
    Los = []
    print(k)
    # p0 = torch.tensor([[k * rt, 0.0]], device="cpu")
    # model = torch.load("model")
    # model.P0 = nn.Parameter(p0, requires_grad=False)
    # model.theh = nn.Parameter(
    #     torch.tensor([torch.arcsin(torch.ones([]) / (-k / 5))]),
    #     requires_grad=True,
    # )
    model.dl = nn.Parameter(torch.tensor(0.2 * rt), requires_grad=False)
    model.to("cpu")
    for i in range(25000):
        opt.zero_grad()
        z = model()
        loss = cost_function(z[0], z[1])
        loss.backward()
        opt.step()
        torch.set_printoptions(precision=4, linewidth=1000)
        print(
            f"{i}\t{z[0][-1]}\t{loss.tolist():.4f}\t{preloss.tolist():.4f}\t{(torch.abs(preloss-loss)).tolist():.4f}"
        )
        # if torch.abs(preloss - loss).tolist() < 1e-20 * rt:
        #     z = model()
        #     plt.gca().set_aspect(1.0)
        #     plt.xlim((-40 * rt, 40 * rt))
        #     plt.ylim((-40 * rt, 40 * rt))
        #     plt.scatter(z[0].detach().cpu()[:, 0], z[0].detach().cpu()[:, 1], s=1)
        #     # print(z[1])
        #     plt.scatter(z[1].detach().cpu()[:, 0], z[1].detach().cpu()[:, 1], s=1)
        #     plt.savefig(f"figs/tmp_{i}.png")
        #     # plt.show()
        #     with open("latestparam", "w") as fo:
        #         for pr in model.parameters():
        #             # print(pr)
        #             fo.write(f"{pr}\n")
        #     break
        preloss = loss
        if (i + 1) % 500 == 0 or i == 0:
            Los.append(loss)
            z = model()
            plt.gca().set_aspect(1.0)
            # plt.xlim((-15 * rt, 40 * rt))
            # plt.ylim((-55 * rt / 2, 55 * rt / 2))
            plt.scatter(Pp[:, 0], Pp[:, 1], s=2)
            plt.scatter(z[0].detach().cpu()[:, 0], z[0].detach().cpu()[:, 1], s=2)
            # print(z[1])
            plt.scatter(z[1].detach().cpu()[:, 0], z[1].detach().cpu()[:, 1], s=2)
            plt.savefig(f"figss/tmp_{i}.png")
            plt.clf()
            # plt.show()
            with open("latestparam", "w") as fo:
                for pr in model.parameters():
                    # print(pr)
                    fo.write(f"{pr}\n")
            torch.save(model.vec, "4vec")
            torch.save(model.vec2, "4vec2")
            torch.save(Los, f"Los4{-20}")
            torch.save(model, f"model4{-20}")

# z = model().detach().cpu()
# print(z, z.device, sep="\n")

# import matplotlib.pyplot as plt

# plt.scatter(z[:, 0], z[:, 1], alpha=1)
# plt.gca().set_aspect(1.0)
# plt.xlim((-15 * rt, 40 * rt))
# plt.ylim((-55 * rt / 2, 55 * rt / 2))
# plt.savefig("figs/tmp.png")
# plt.show()
