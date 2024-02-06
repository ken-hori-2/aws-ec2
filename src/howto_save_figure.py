

import matplotlib.pyplot as plt
import numpy as np

# epochs = np.linspace(0, 10, 1) # range(len(accs))
epochs = np.arange(10)

# # サイズが6,3の描画領域と1行2列のグラフの描画領域を作る
# fig, ax = plt.subplots(nrows=1, ncols=2) # , figsize=(16,24))
# # figureのタイトルを設定する
# fig.suptitle("loss & acc")
# plt.style.use("ggplot")
# ax[0].plot(epochs, epochs * 2, label="train loss", color = "orange")
# ax[1].plot(epochs, epochs * -2, label="accurucy", color = "green")
# ax[0].legend()
# ax[1].legend()
# plt.savefig('test.png')
# # ax.close()




fig = plt.figure(figsize=(15, 8))

#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
fig.suptitle("loss & acc")
plt.style.use("ggplot")
ax1.plot(epochs, epochs * 2, label="train loss", color = "orange")
ax2.plot(epochs, epochs * -2, label="accurucy", color = "green")
plt.legend()
plt.savefig('test.png')

print(epochs, epochs*2)