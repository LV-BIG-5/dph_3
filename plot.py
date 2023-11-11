import matplotlib.pyplot as plt

res = []
with open("./results20231102-123952.txt", "r") as f:
    for line in f.readlines():
        if line.startswith("mean"):
            res.append(float(line.split(":")[1][1:]))
new_res = []
for i in range(len(res)):
    if i % 20 == 0:
        new_res.append(res[i]*8/8)
plt.plot(list(range(len(new_res))), new_res, 'ro-')
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.savefig("acc.png", dpi = 1000)