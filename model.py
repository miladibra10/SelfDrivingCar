import argparse
import os
import pandas
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from utils import batch_generator
from .settings import INPUT_SHAPE


def load_data(args):
    """
    to load data from csv
    """
    data = pandas.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    x = data[['center', 'left', 'right']].values
    y = data['steering'].values

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=args.test_size, random_state=0)
    return train_x, valid_x, train_y, valid_y


def get_model(args):
    """
    model implementation
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-0.1, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model


def learn_model(model, args, train_x, valid_x, train_y, valid_y):
    """
    learn model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, train_x, train_y, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, valid_x, valid_y, args.batch_size, False),
                        nb_val_samples=len(valid_x),
                        callbacks=[checkpoint],
                        verbose=1)


def boolean_arg_getter(s):
    s = s.lower()
    return s == 'true' or s == 'yes'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model for Self Driving Car')
    parser.add_argument('-d', help='Data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='Test Size', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='Dropout Probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='Epochs number', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='Samples per Epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='Batch size', dest='batch_size', type=int, default=40)
    parser.add_argument('-o', help='Save best models only', dest='save_best_only', type=bool, default=True)
    parser.add_argument('-l', help='Learning rate', dest='learning_rate', type=float, default=1**-4)
    args = parser.parse_args()

    data = load_data(args)
    model = get_model(args)
    learn_model(model, args, *data)