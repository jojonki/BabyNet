import  numpy as np

from dataset import spiral
from common.layers import TwoLayerNet
from common.optimizers import SGD
from common.trainer import Trainer


def main():
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    output_size = 3
    learning_rate = 1.0

    x, t = spiral.load_data()
    model = TwoLayerNet(2, hidden_size, output_size)
    optimizer = SGD(learning_rate)
    trainer = Trainer(model, optimizer)
    trainer.fit(x, t, max_epoch, batch_size)
    trainer.plot()




if __name__ == '__main__':
    main()
