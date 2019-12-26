import  numpy as np
from nets.two_layer_net import TwoLayerNet


def main():
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    out = model.predict(x)
    print(out)


if __name__ == '__main__':
    main()
