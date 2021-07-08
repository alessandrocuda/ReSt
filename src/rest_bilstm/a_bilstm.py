import tensorflow as tf

class Attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
    
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                           initializer="glorot_uniform", trainable=True)
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                           initializer="zeros", trainable=True)
    
        super(Attention, self).build(input_shape)

    def call(self, x):
    
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
    
        if self.return_sequences:
            return a, output
    
        return a, tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences 
        })
        return config

def a_bilstm(input_shape, input_shape_pos, bi_units = 256, bi_dropout = 0.5, dropout = 0.5, hn=128, lr=0.01, print_model=False):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Bidirectional, LSTM, Embedding
    from tensorflow.keras.layers import Dense, Input, Flatten, Activation
    from tensorflow.keras.layers import Concatenate

    """ HyperParameters """

    text_seq_input = Input(shape=input_shape, name="text")
    pos_seq_input = Input(shape=input_shape_pos, name="pos")
    pos_emb = Embedding(17, 16, input_length=input_shape_pos[0])(pos_seq_input)

    input_bi    = Concatenate(axis=2)([text_seq_input, pos_emb])

    extra_feature = Input(shape=(8,), name="extra")

    sentence_encoder = Bidirectional(LSTM(bi_units, dropout = bi_dropout,  return_sequences=True))(input_bi)
    sentence_attention = Attention(return_sequences=False)(sentence_encoder)
    
    input_ff    = Concatenate(axis=1)([sentence_attention, extra_feature])
    l_drop      = Dropout(dropout)(input_ff)
    l_hidden    = Dense(hn, activation='relu')(input_ff)
    l_drop      = Dropout(dropout)(l_hidden)
    l_out_st    = Dense(1, activation='sigmoid', name="st")(l_drop)  #dims output

    model_cnn   = Model(inputs=[text_seq_input, pos_seq_input, extra_feature], outputs=l_out_st)
    if print_model:
        model_cnn.summary()
        tf.keras.utils.plot_model(model_cnn, "my_first_model.png", show_shapes=True)
    
    model_cnn.compile(
        loss= tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False),
    )

    return model_cnn

def get_score(X_train, Y_train, X_val_es, Y_val_es, X_val, Y_val,  hyper_param):
    from src.utils.callbacks import ReturnBestEarlyStopping
    import numpy as np
    from sklearn.metrics import f1_score

    input_shape_text = (X_train["text"][0].shape[0], X_train["text"][0].shape[1],)
    input_shape_pos = (X_train["pos"][0].shape[0], )
    model = a_bilstm(input_shape_text, input_shape_pos, **hyper_param)
    
    best_callback = ReturnBestEarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min", restore_best_weights=True)
    history = model.fit(X_train, Y_train, batch_size=64, epochs=50, validation_data=(X_val_es, Y_val_es), callbacks=[best_callback], verbose = 0)

    #evaluation
    y_pred = np.where(model.predict(X_train) >0.5, 1,0)
    f1      = f1_score(Y_train, y_pred, average="macro")
    y_pred = np.where(model.predict(X_val) >0.5,1,0)
    val_f1  = f1_score(Y_val, y_pred, average="macro")
    loss = model.evaluate(X_train, Y_train, verbose = 0)
    val_loss = model.evaluate(X_val, Y_val, verbose = 0)

    return {"loss": loss, "val_loss": val_loss, "f1_macro": f1, "val_f1_macro": val_f1}
    