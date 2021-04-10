def bilstm_text_pos_extra(input_shape_text, input_shape_pos, bi_units = 256, bi_dropout = 0.5, hn=128, dropout = 0.5, lr=0.01, print_model=False):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Bidirectional, LSTM, Embedding
    from tensorflow.keras.layers import Dense, Input, Flatten, Activation
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.regularizers import l2

    """ HyperParameters """

    text_seq_input = Input(shape=input_shape_text, name="text")
    pos_seq_input = Input(shape=input_shape_pos, name="pos")
    pos_emb = Embedding(17, 16, input_length=input_shape_pos[0])(pos_seq_input)
    extra_feature = Input(shape=(8,), name="extra")

    sentence_encoder = Bidirectional(LSTM(bi_units, return_sequences=False))(text_seq_input)
    
    pos_encoder = Bidirectional(LSTM(bi_units, return_sequences=False))(pos_emb)

    input_ff    = Concatenate(axis=1)([sentence_encoder, pos_encoder, extra_feature])
    l_drop      = Dropout(dropout)(input_ff)
    l_hidden    = Dense(hn, activation='relu')(l_drop)
    l_drop      = Dropout(dropout)(l_hidden)
    l_out_st    = Dense(1, activation='sigmoid', name="st")(l_drop)  #dims output

    model_cnn   = Model(inputs=[text_seq_input, pos_seq_input, extra_feature], outputs=l_out_st)
    if print_model:
        model_cnn.summary()
        tf.keras.utils.plot_model(model_cnn, "my_first_model.png", show_shapes=True)
    
    model_cnn.compile(
        loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False),
        metrics= [f1_macro]
    )

    return model_cnn

def get_score(X_train, Y_train, X_val_es, Y_val_es, Y_val,  hyper_param):
    input_shape_text = (X_train["text"][0].shape[0], X_train["text"][0].shape[1],)
    input_shape_pos = (X_train["pos"][0].shape[0], )
    model = bilstm_text_pos_extra(input_shape_text, input_shape_pos, **hyper_param)
    best_callback = ReturnBestEarlyStopping(monitor="val_f1_macro", min_delta=0, verbose=1, mode="max", restore_best_weights=True)
    history = model.fit(X_train, Y_train, batch_size=64, epochs=20, validation_data=(X_val_es, Y_val_es), callbacks=[best_callback], verbose = 0)

    y_pred = np.where(model.predict(X_train) > 0.5, 1,0)
    f1      = f1_score(Y_train, y_pred, average="macro")
    y_pred = np.where(model.predict(X_val) > 0.5, 1,0)
    val_f1  = f1_score(Y_val, y_pred, average="macro")

    folds_f1.append(val_f1)
    folds_result[name_fold] = {"f1": f1, "val_f1": val_f1}
    return folds_result