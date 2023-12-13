from datetime import datetime
from keras import backend as K


def time_to_int(time):
    """
    Convert time to int
    """

    timestamp = int(round(time.timestamp()))
    return timestamp


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def r2metrics(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_reg = K.sum(K.square(y_true - K.mean(y_true)))
    SS_tot = SS_res + SS_reg
    return (1 - SS_res/(SS_tot + K.epsilon()))
