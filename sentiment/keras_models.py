from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
from keras.layers import merge
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, Flatten, \
    TimeDistributed, RepeatVector, Permute, MaxoutDense, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import preprocessing

from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.topology import Layer


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights

    Returns:

    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class MeanOverTime(Layer):
    """
    Layer that computes the mean of timesteps returned from an RNN and supports masking
    Example:
        activations = LSTM(64, return_sequences=True)(words)
        mean = MeanOverTime()(activations)
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            if not K.any(mask):
                return K.mean(x, axis=1)
            else:
                return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True),
                              K.floatx())
        else:
            return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_mask(self, input, input_mask=None):
        return None


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer='zero',
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]



def embeddings_layer(max_length, embeddings, trainable=False, masking=False,
                     scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding


def weighted_states(activations, rnn_size, input_length, attention="single"):
    if attention == "all":
        attention = Flatten()(activations)
        attention = Dense(input_length, activation='tanh')(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(rnn_size)(attention)
        attention = Permute([2, 1])(attention)
        return merge([activations, attention], mode='mul')
    elif attention == "single":
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        # attention = Dense(1, activation='tanh')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(rnn_size)(attention)
        attention = Permute([2, 1])(attention)
        return merge([activations, attention], mode='mul')


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def get_LSTM(cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = LSTM( cells, return_sequences=return_sequences, kernel_regularizer=l2(l2_reg), 
                recurrent_dropout=dropout_U, implementation=1)
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn


def build_attention_RNN(embeddings, classes, max_length, unit=LSTM, cells=64,
                        layers=1, **kwargs):
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)

    model = Sequential()
    model.add(embeddings_layer(max_length=max_length, embeddings=embeddings,
                               trainable=False, masking=True, scale=False,
                               normalize=False))

    if noise > 0:
        model.add(GaussianNoise(noise))
    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        # model.add(get_RNN(unit, cells, bi, return_sequences=rs, dropout_U=dropout_rnn_U))
        model.add(get_LSTM(cells, bi, return_sequences=rs, dropout_U=dropout_rnn_U))
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))

    if attention == "memory":
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == "simple":
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        # model.add(Highway())
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy')
    return model


def target_RNN(wv, tweet_max_length, aspect_max_length, classes=2, **kwargs):
    ######################################################
    # HyperParameters
    ######################################################
    noise = kwargs.get("noise", 0)
    trainable = kwargs.get("trainable", False)
    rnn_size = kwargs.get("rnn_size", 75)
    rnn_type = kwargs.get("rnn_type", LSTM)
    final_size = kwargs.get("final_size", 100)
    final_type = kwargs.get("final_type", "linear")
    use_final = kwargs.get("use_final", False)
    drop_text_input = kwargs.get("drop_text_input", 0.)
    drop_text_rnn = kwargs.get("drop_text_rnn", 0.)
    drop_text_rnn_U = kwargs.get("drop_text_rnn_U", 0.)
    drop_target_rnn = kwargs.get("drop_target_rnn", 0.)
    drop_rep = kwargs.get("drop_rep", 0.)
    drop_final = kwargs.get("drop_final", 0.)
    activity_l2 = kwargs.get("activity_l2", 0.)
    clipnorm = kwargs.get("clipnorm", 5)
    bi = kwargs.get("bi", False)
    lr = kwargs.get("lr", 0.001)

    attention = kwargs.get("attention", "simple")
    #####################################################
    shared_RNN = get_RNN(rnn_type, rnn_size, bi=bi, return_sequences=True,
                         dropout_U=drop_text_rnn_U)

    input_tweet = Input(shape=[tweet_max_length], dtype='int32')
    input_aspect = Input(shape=[aspect_max_length], dtype='int32')

    # Embeddings
    tweets_emb = embeddings_layer(max_length=tweet_max_length, embeddings=wv,
                                  trainable=trainable, masking=True)(
        input_tweet)
    tweets_emb = GaussianNoise(noise)(tweets_emb)
    tweets_emb = Dropout(drop_text_input)(tweets_emb)

    aspects_emb = embeddings_layer(max_length=aspect_max_length, embeddings=wv,
                                   trainable=trainable, masking=True)(
        input_aspect)
    aspects_emb = GaussianNoise(noise)(aspects_emb)

    # Recurrent NN
    h_tweets = shared_RNN(tweets_emb)
    h_tweets = Dropout(drop_text_rnn)(h_tweets)

    h_aspects = shared_RNN(aspects_emb)
    h_aspects = Dropout(drop_target_rnn)(h_aspects)
    h_aspects = MeanOverTime()(h_aspects)
    h_aspects = RepeatVector(tweet_max_length)(h_aspects)

    # Merge of Aspect + Tweet
    representation = merge([h_tweets, h_aspects], mode='concat')

    # apply attention over the hidden outputs of the RNN's
    att_layer = AttentionWithContext if attention == "context" else Attention
    representation = att_layer()(representation)
    representation = Dropout(drop_rep)(representation)

    if use_final:
        if final_type == "maxout":
            representation = MaxoutDense(final_size)(representation)
        else:
            representation = Dense(final_size, activation=final_type)(
                representation)
        representation = Dropout(drop_final)(representation)

    ######################################################
    # Probabilities
    ######################################################
    probabilities = Dense(1 if classes == 2 else classes,
                          activation="sigmoid" if classes == 2 else "softmax",
                          activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=[input_aspect, input_tweet], output=probabilities)

    loss = "binary_crossentropy" if classes == 2 else "categorical_crossentropy"
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss=loss)
    return model


def cnn_simple(wv, sent_length, **params):
    model = Sequential()
    model.add(
        embeddings_layer(max_length=sent_length, embeddings=wv, masking=False))

    # model.add(Convolution1D(nb_filter=80, filter_length=4, border_mode='valid', activation='relu'))
    model.add(Conv1D(activation="relu", filters=80, kernel_size=4, padding="valid"))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer="adam", loss='categorical_crossentropy')
    return model


def cnn_multi_filters(wv, sent_length, nfilters, nb_filters, **kwargs):
    noise = kwargs.get("noise", 0)
    trainable = kwargs.get("trainable", False)
    drop_text_input = kwargs.get("drop_text_input", 0.)
    drop_conv = kwargs.get("drop_conv", 0.)
    activity_l2 = kwargs.get("activity_l2", 0.)

    input_text = Input(shape=(sent_length,), dtype='int32')

    emb_text = embeddings_layer(max_length=sent_length, embeddings=wv,
                                trainable=trainable, masking=False)(input_text)
    emb_text = GaussianNoise(noise)(emb_text)
    emb_text = Dropout(drop_text_input)(emb_text)

    pooling_reps = []
    for i in nfilters:
        feat_maps = Convolution1D(nb_filter=nb_filters,
                                  filter_length=i,
                                  border_mode="valid",
                                  activation="relu",
                                  subsample_length=1)(emb_text)
        pool_vecs = MaxPooling1D(pool_length=2)(feat_maps)
        pool_vecs = Flatten()(pool_vecs)
        # pool_vecs = GlobalMaxPooling1D()(feat_maps)
        pooling_reps.append(pool_vecs)

    representation = merge(pooling_reps, mode='concat')

    representation = Dropout(drop_conv)(representation)

    probabilities = Dense(3, activation='softmax',
                          activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=input_text, output=probabilities)
    model.compile(optimizer="adam", loss='categorical_crossentropy')

    return model
