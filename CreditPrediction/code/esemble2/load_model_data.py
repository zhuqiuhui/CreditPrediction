import pandas as pd


def load_train():
    train_xy = pd.read_csv('data/train_x_oneHot.csv')
    train_xy = train_xy.drop(['uid'], axis=1)
    return train_xy


def load_test():
    test = pd.read_csv('data/test_x_oneHot.csv')
    test_uid = test.uid
    test_x = test.drop(['uid'], axis=1)
    return test_x, test_uid


def test():
    pass


if __name__ == '__main__':
    test()
