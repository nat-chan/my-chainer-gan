import train
import numpy as np
import matplotlib.pyplot as plt
from dcgan.net import Discriminator, Generator
from chainer import serializers
from common.evaluation import sample_generate
from chainer import Variable
from PIL import Image

model = Generator()
serializers.load_npz('result_dcgan64/DCGANGenerator_100000.npz', model)


# ダックタイピング
# sample_generate(model, "tmp")(type('',(),{'updater':type('',(),{'iteration':'test'})()})())

#rows, colsの順
# 0 1 2 3 
# 4 5 6 7
# 8 9 ...

def make_hidden(self, batchsize):
    #                        [low, high) 
    return np.random.uniform(-4, 1, size=(batchsize, self.n_hidden, 1, 1)).astype(np.float32)

def make_image(model, rows=1, cols=1, seed=0):
        np.random.seed(seed)
        z = Variable(np.asarray(model.make_hidden(rows*cols)))
#        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
#            x = model(z)
        x = model(z).data
#        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        _, _, h, w = x.shape
        x = x.reshape((rows, cols, 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * h, cols * w, 3))

        Image.fromarray(x).show()


def predict(z):
    """
    学習済み生成モデルを使って推論する
    """
    return model(z).data[0].transpose(1,2,0)

def linear_map0_1(x):
    """
    値域を区間[0,1]に押し込める
    """
    _max = np.max(x)
    _min = np.min(x)
    return (x - _min)/(_max - _min)

def main():
#    z = np.array([[[0.1]*128]], dtype=np.float32)
    z = np.array([[list(i/128.0 for i in range(128))]], dtype=np.float32)
    pred = predict(z)
    plt.imshow(pred)
    plt.show()

def anime():
    import matplotlib.animation as animation
    fig = plt.figure()
    ims = []
    for i in range(10):
        z = np.array([[[0.1*i]*128]], dtype=np.float32)
        pred = predict(z)
        im = plt.imshow(pred, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()

if __name__ == '__main__':
    main()
#    anime()
