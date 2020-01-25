from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Lambda
from keras.layers import  Convolution2D, AveragePooling2D, ZeroPadding2D, merge, Reshape, Activation, LeakyReLU
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras.models import Model
from keras.layers.advanced_activations import PReLU

def build_vgg(dense_size=256, dropout_val=0.2):
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layers
    model.add(Flatten())

    model.add(Dropout(dropout_val))
    model.add(Dense(dense_size, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(LeakyReLU(alpha=0.05))


    model.add(Dropout(dropout_val))
    model.add(Dense(dense_size, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(10, activation='softmax'))
    
    return model



def build_cifar_model(num_classes, x_shape, weight_decay=0.005, dense_size=512, conv_dropout=0.4, dense_dropout=0.5):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dense_dropout))

    model.add(Flatten())
    model.add(Dense(dense_size,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(dense_dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def functional_cifar(num_classes, input_shape, weight_decay=0.01, dense_size=512, conv_dropout=0.0, dense_dropout=0.0):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    
    inpt = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inpt)
    
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    #x = Activation('relu')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    #x = Activation('relu')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    #x = Activation('relu')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dense_dropout)(x)

    x = Flatten()(x)
    x = Dense(dense_size,kernel_regularizer=regularizers.l2(1e-3))(x)
    #x = Activation('linear')(x)
    x = PReLU()(x)
    
    x = Lambda(lambda i: K.l2_normalize(i,axis=-1))(x)

    return Model(inpt, x)

def functional_end(num_classes, input_shape, weight_decay=0.0002, dense_size=512, conv_dropout=0.0, dense_dropout=0.0):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    
    inpt = Input(shape=input_shape)
    
    #x = Flatten()(inpt)
    
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inpt)
    
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dense_dropout)(x)

    x = Flatten()(x)
    x = Dense(dense_size,kernel_regularizer=regularizers.l2(1e-3))(x)
    x = Activation('linear')(x)
    
    x = Lambda(lambda i: K.l2_normalize(i,axis=-1))(x)
    
    x = Dense(10, activation='softmax')(x)

    return Model(inpt, x)