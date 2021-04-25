from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import add, Input, Conv2D, Conv1D, Activation, Flatten, Dense, MaxPool2D
from keras.optimizers import adam
import attention, we
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
from keras.models import load_model
import time


class cnn():
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.flg = 0


    def ResBlock(self,x, filters, kernel_size, dilation_rate):
        r = Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                   activation='relu', kernel_initializer='he_normal', strides=(2, 2))(x)
        r = Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                   activation='relu', kernel_initializer='he_normal', strides=(2, 2))(r)
        if x.shape[-1] == filters:
            shortcut = x
        else:
            shortcut = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(x)  # shortcut
        o = add([r, shortcut])
        o = Activation('relu')(o)
        o = MaxPool2D((2, 2))(o)
        return o


    def cnn_att(self):
        inputs = Input(shape=(256, 256, 3))
        x = self.ResBlock(inputs, filters=16, kernel_size=(3, 3), dilation_rate=1)
        x = attention.Attention(attention_activation='relu')(x)
        x = self.ResBlock(x, filters=32, kernel_size=(3, 3), dilation_rate=2)
        x = self.ResBlock(x, filters=64, kernel_size=(3, 3), dilation_rate=4)
        x = Flatten(name='feature')(x)
        x = Conv1D(1024, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv1D(512, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                  kernel_initializer='he_normal')(x)

        model = Model(input=inputs, output=x)
        # View the network structure
        model.summary()
        # Compiling model
        opt = adam(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
        return model


    def train(self, path):
        time1 = time.time()
        train_x, train_y = we.without_we(path, 'train')
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
        self.cnn = self.cnn_att()
        self.cnn.fit(train_x, train_y, batch_size=64, epochs=150, validation_data=(test_x, test_y))

        test_x_, test_y_ = we.without_we('./data/test/', 'test')
        self.cnn.evaluate(test_x_, test_y_)
        pre_y = self.cnn.predict(test_x_)

        print('accuracy: {}'.format(accuracy_score(pre_y, test_y_)))
        print('f1score: {}'.format(f1_score(pre_y, test_y_, average='micro')))
        print('mcc: {}'.format(matthews_corrcoef(pre_y, test_y_)))
        time2 = time.time()
        print('time', time2 - time1)

        self.cnn.save('./out/model/model.h5')

    def test(self, path):
        test_x, test_y = we.without_we(path, 'test')
        self.cnn = load_model("model.h5")
        self.cnn.evaluate(test_x, test_y)
        pre_y = self.cnn.predict(test_x)

        print('accuracy: {}'.format(accuracy_score(pre_y, test_y)))
        print('f1score: {}'.format(f1_score(pre_y, test_y, average='micro')))
        print('mcc: {}'.format(matthews_corrcoef(pre_y, test_y)))


if __name__ == '__main__':
    path_train = './data/train/'
    path_test = './data/test/'
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
    process = cnn()
    process.train(path=path_train)
    # process.test(path=path_test)