from result_dcgan64_selebA.log import l
import matplotlib.pyplot as plt
import numpy as np

dl = [k['loss_gen'] for k in l]
ep = np.array([k['epoch'] for k in l])
d = [0]+list(np.where(ep[:-1] != ep[1:])[0] + 1)
d = d[::10] + [999]

plt.plot(dl)
plt.xticks(d, [ep[i] for i in d])
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.title('Discriminator(DCGAN SelebA)')
plt.title('Generator(DCGAN SelebA)')
plt.show()
