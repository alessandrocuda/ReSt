def kim_cnn_pos(input_shape, input_shape_pos, filters = 256, filter_sizes = [2,4,6], dropout = 0.5, hn=128, lr=0.01, print_model=False):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D
    from tensorflow.keras.layers import Dense, Input, Flatten, Activation
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.regularizers import l2

    """ HyperParameters """

    text_seq_input = Input(shape=input_shape, name="text")
    pos_seq_input = Input(shape=input_shape_pos, name="pos")
    extra_feature = Input(shape=(8,), name="extra")

    convs = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=filters, kernel_size=filter_size)(text_seq_input)
        l_relu = Activation("relu")(l_conv)
        l_pool = GlobalMaxPool1D()(l_relu)   
        convs.append(l_pool)
    
    convs_pos = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=filters, kernel_size=filter_size)(pos_seq_input)
        l_relu = Activation("relu")(l_conv)
        l_pool = GlobalMaxPool1D()(l_relu)   
        convs_pos.append(l_pool)

    l_merge     = Concatenate(axis=1)(convs)
    l_flat      = Flatten()(l_merge)
    l_merge_pos = Concatenate(axis=1)(convs_pos)
    l_flat_pos  = Flatten()(l_merge_pos)

    input_ff    = Concatenate(axis=1)([l_flat, l_flat_pos, extra_feature])
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
    )

    return model_cnn

def get_score(X_train, Y_train, X_val_es, Y_val_es, X_val, Y_val,  hyper_param):
    from src.utils.callbacks import ReturnBestEarlyStopping
    import numpy as np
    from sklearn.metrics import f1_score

    input_shape_text = (X_train["text"][0].shape[0], X_train["text"][0].shape[1],)
    input_shape_pos = (X_train["pos"][0].shape[0], X_train["pos"][0].shape[1] )
    model = kim_cnn_pos(input_shape_text, input_shape_pos, **hyper_param)
    
    best_callback = ReturnBestEarlyStopping(monitor="val_loss", patience=50, verbose=0, mode="min", restore_best_weights=True)
    history = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_data=(X_val_es, Y_val_es), callbacks=[best_callback], verbose = 0)
        
    #evaluation
    y_pred = np.where(model.predict(X_train) >0.5, 1,0)
    f1      = f1_score(Y_train, y_pred, average="macro")
    y_pred = np.where(model.predict(X_val) >0.5,1,0)
    val_f1  = f1_score(Y_val, y_pred, average="macro")
    loss = model.evaluate(X_train, Y_train, verbose = 0)
    val_loss = model.evaluate(X_val, Y_val, verbose = 0)

    return {"loss": loss, "val_loss": val_loss, "f1_macro": f1, "val_f1_macro": val_f1}