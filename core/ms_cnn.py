from keras.models import Model
from keras.layers import add, Input, Conv2D, Conv1D, Activation, Flatten, Dense, MaxPool2D
from keras.optimizers import adam
import attention, we
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
from keras.models import load_model
import time


class mscnn():
    def __init__(self):
        self.input_shape = (256, 256, 1)


    def cnn_1(self):
        inputs = Input(shape=(256, 256, 1))
        x = Conv2D(16, (7, 7), padding='same', activation='relu', strides=(2, 2))(inputs)
        x = MaxPool2D((2, 2))(x)
        x = Conv2D(32, (5, 5), padding='same', strides=(2, 2))(x)
        x = Conv2D(32, (5, 5), padding='same', activation='relu', strides=(2, 2))(x)
        x = MaxPool2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(x)
        x = MaxPool2D((2, 2))(x)
        x = Flatten(name='Feature')(x)
        self.feature_shape = K.shape(x).eval()
        x =Dense(512)(x)
        x =Dense(512)(x)
        x =Dense(6, activation='softmax')(x)

        model = Model(input=inputs, output=x)
        # View the network structure
        model.summary()
        # Compiling model
        opt = adam(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
        return model


    def cnn_ms(self):
        inputs = Input(shape=self.feature_shape)
        x = Dense(512)(inputs)
        x = Dense(512)(x)
        x = Dense(6, activation='softmax')(x)

        model = Model(input=inputs, output=x)
        # View the network structure
        model.summary()
        # Compiling model
        opt = adam(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
        return model


    def train(self, path):
        time1 = time.time()
        train_x_1, train_x_2, train_x_3, train_y,  = we.with_rgb(path, 'train')

        train_x_1, train_y_1, test_x_1, test_y_1 = train_test_split(train_x_1, train_y, test_size=0.2, random_state=42)
        train_x_2, train_y_2, test_x_2, test_y_2 = train_test_split(train_x_2, train_y, test_size=0.2, random_state=42)
        train_x_3, train_y_3, test_x_3, test_y_3 = train_test_split(train_x_3, train_y, test_size=0.2, random_state=42)
        self.cnn = self.cnn_1()
        cnn_1_feature = keras.Model(self.cnn.input, self.cnn.output)
        cnn_2_feature = keras.Model(self.cnn.input, self.cnn.output)
        cnn_3_feature = keras.Model(self.cnn.input, self.cnn.output)
        cnn_1_feature.fit(train_x_1, train_y_1, batch_size=32, epochs=150, validation_data=(test_x_1, test_y_1))
        cnn_2_feature.fit(train_x_2, train_y_2, batch_size=32, epochs=150, validation_data=(test_x_2, test_y_2))
        cnn_3_feature.fit(train_x_3, train_y_3, batch_size=32, epochs=150, validation_data=(test_x_3, test_y_3))
        cnn_1_feature.save('./out/model/model_1.h5')
        cnn_2_feature.save('./out/model/model_2.h5')
        cnn_2_feature.save('./out/model/model_3.h5')
        x = cnn_1_feature.get_layer('feature').output
        y = cnn_2_feature.get_layer('feature').output
        z = cnn_2_feature.get_layer('feature').output
        inputs = add([x, y])
        inputs = add([inputs, z])
        self.multi_cnn = self.cnn_ms()
        self.multi_cnn.fit(inputs,train_y, batch_size=64, epochs=150)
        self.cnn.save('./out/model/model_0.h5')

        test_x_1, test_x_2, test_x_3, test_y = we.with_we_r('./data/test/', 'test')
        a = cnn_1_feature.predict(test_x_1)
        b = cnn_2_feature.predict(test_x_2)
        c = cnn_3_feature.predict(test_x_3)
        test_x = add([a, b])
        test_x = add([test_x, c])
        pre_y = self.multi_cnn.predict(test_x)
        print('accuracy: {}'.format(accuracy_score(pre_y, test_y)))
        print('f1score: {}'.format(f1_score(pre_y, test_y, average='micro')))
        print('mcc: {}'.format(matthews_corrcoef(pre_y, test_y)))
        time2 = time.time()
        print('time', time2-time1)

    def test_mutil_cnn(self, path):
        test_x_1, test_x_2, test_x_3, test_y = we.with_we_r(path, 'test')
        cnn_1_feature = load_model('./out/model/model_1.h5')
        cnn_2_feature = load_model('./out/model/model_2.h5')
        cnn_3_feature = load_model('./out/model/model_2.h5')
        self.cnn = load_model("./out/model/model_0.h5")
        a = cnn_1_feature.predict(test_x_1)
        b = cnn_2_feature.predict(test_x_2)
        c = cnn_3_feature.predict(test_x_3)
        test_x = add([a, b])
        test_x = add([test_x, c])
        pre_y = self.multi_cnn.predict(test_x)

        print('accuracy: {}'.format(accuracy_score(pre_y, test_y)))
        print('f1score: {}'.format(f1_score(pre_y, test_y, average='micro')))
        print('mcc: {}'.format(matthews_corrcoef(pre_y, test_y)))


if __name__ == '__main__':
    path_train = './data/train/'
    path_test = './data/test/'
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
    process = mscnn()
    process.train(path=path_train)
    # process.test_mutil_cnn(path=path_test)