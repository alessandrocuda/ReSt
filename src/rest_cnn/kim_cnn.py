def kim_cnn(input_shape, filters = 256, filter_sizes = [2,4,6], dropout_cnn=0.5, dropout_ff=0.5 , print_model=False):
    from tensorflow import tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D
    from tensorflow.keras.layers import Dense, Input, Flatten, Activation
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.regularizers import l2

    """ HyperParameters """

    text_seq_input = Input(shape=input_shape, name="text")

    convs = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=filters, kernel_size=filter_size)(text_seq_input)
        l_relu = Activation("relu")(l_conv)
        l_drop = Dropout(dropout_cnn)(l_relu)        
        l_pool = GlobalMaxPool1D()(l_drop)   
        convs.append(l_pool)

    l_merge     = Concatenate(axis=1)(convs)

    l_flat      = Flatten()(l_merge)
    l_drop      = Dropout(dropout_ff)(l_flat)
    l_hidden    = Dense(128, activation='relu')(l_drop)
    l_drop      = Dropout(dropout_ff)(l_hidden)
    l_out_st    = Dense(1, activation='sigmoid', name="st")(l_drop)  #dims output

    model_cnn   = Model(inputs=text_seq_input, outputs=l_out_st)
    if print_model:
        model_cnn.summary()
        tf.keras.utils.plot_model(model_cnn, "my_first_model.png", show_shapes=True)
    return model_cnn