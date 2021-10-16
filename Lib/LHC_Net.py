from abc import ABC
import tensorflow as tf


print(tf.__version__)


# resolution^2 deve essere divisibile per num_heads
class LHC_Module(tf.keras.layers.Layer):
    def __init__(self, pool_size, head_emb_dim, num_heads, num_channels, resolution, kernel_size, norm_c, name):
        super(LHC_Module, self).__init__()
        self.pool_size = pool_size
        self.head_emb_dim = head_emb_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.norm_c = norm_c

        self.Poolq = tf.keras.layers.AvgPool2D(pool_size=(self.pool_size, self.pool_size),
                                               strides=(1, 1),
                                               padding='same')
        self.Poolk = tf.keras.layers.MaxPool2D(pool_size=(self.pool_size, self.pool_size),
                                               strides=(1, 1),
                                               padding='same')

        self.Wqk = [tf.keras.layers.Dense(units=self.head_emb_dim, activation='linear') for _ in range(self.num_heads)]

        self.Wp = tf.keras.layers.Dense(units=self.num_channels, activation='sigmoid')

        self.Wv = tf.keras.layers.Conv2D(filters=self.num_channels,
                                         kernel_size=self.kernel_size,
                                         strides=(1, 1),
                                         padding='same',
                                         activation='linear')
        self.Poolv = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                               strides=(1, 1),
                                               padding='same')

        self.sum = tf.keras.layers.Add()

        self.Name_1_ = 'LHC_1_'+name

    def VectScaledDotProdAttention(self, query, key, value):
        scores = tf.linalg.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, num_channels, num_channels)
        scores_p = tf.math.reduce_mean(scores, axis=3)  # (batch_size, num_heads, num_channels)
        scores_p = self.Wp(scores_p)  # (batch_size, num_heads, num_channels)
        scores_p = tf.expand_dims(scores_p, axis=-1)  # (batch_size, num_heads, num_channels, 1)
        norm_scores = tf.math.divide(scores, tf.math.pow(tf.dtypes.cast(key.shape[3], tf.float32), self.norm_c + scores_p))  # (batch_size, num_heads, num_channels, num_channels)
        weights = tf.nn.softmax(norm_scores, axis=3)  # (batch_size, num_heads, num_channels, num_channels)
        attentions = tf.linalg.matmul(weights, value)  # (batch_size, num_heads, num_channels, head_res_dim)
        return attentions

    def call(self, x):
        batch_size = tf.shape(x)[0]
        num_channels = self.num_channels
        resolution = self.resolution
        head_res_dim = (resolution * resolution) // self.num_heads

        query = x  # (batch_size, resolution, resolution, num_channels)
        query = self.Poolq(query)  # (batch_size, resolution, resolution, num_channels)
        query = tf.reshape(query, shape=(batch_size, resolution * resolution, num_channels))  # (batch_size, resolution^2, num_channels)
        query = tf.transpose(query, perm=[0, 2, 1])  # (batch_size, num_channels, resolution^2)
        query = tf.reshape(query, shape=(batch_size, num_channels, self.num_heads, head_res_dim))  # (batch_size, num_channels, num_heads, head_res_dim)
        query = tf.transpose(query, perm=[0, 2, 1, 3])  # (batch_size, num_heads, num_channels, head_res_dim)
        q = [None] * self.num_heads
        for i in range(self.num_heads):
            q[i] = self.Wqk[i](query[:, i, :, :])  # (batch_size, num_channels, head_emb_dim)
            q[i] = tf.expand_dims(q[i], axis=1)  # (batch_size, 1, num_channels, head_emb_dim)
        query = tf.concat(q, axis=1)  # (batch_size, num_heads, num_channels, head_emb_dim)

        key = x  # (batch_size, resolution, resolution, num_channels)
        key = self.Poolk(key)  # (batch_size, resolution, resolution, num_channels)
        key = tf.reshape(key, shape=(batch_size, resolution * resolution, num_channels))  # (batch_size, resolution^2, num_channels)
        key = tf.transpose(key, perm=[0, 2, 1])  # (batch_size, num_channels, resolution^2)
        key = tf.reshape(key, shape=(batch_size, num_channels, self.num_heads, head_res_dim))  # (batch_size, num_channels, num_heads, head_res_dim)
        key = tf.transpose(key, perm=[0, 2, 1, 3])  # (batch_size, num_heads, num_channels, head_res_dim)
        k = [None] * self.num_heads
        for i in range(self.num_heads):
            k[i] = self.Wqk[i](key[:, i, :, :])  # (batch_size, num_channels, head_emb_dim)
            k[i] = tf.expand_dims(k[i], axis=1)  # (batch_size, 1, num_channels, head_emb_dim)
        key = tf.concat(k, axis=1)  # (batch_size, num_heads, num_channels, head_emb_dim)

        value = self.Wv(x)  # (batch_size, resolution, resolution, num_channels)
        value = self.Poolv(value)  # (batch_size, resolution, resolution, num_channels)
        value = tf.reshape(value, shape=(batch_size, resolution * resolution, num_channels))  # (batch_size, resolution^2, num_channels)
        value = tf.transpose(value, perm=[0, 2, 1])  # (batch_size, num_channels, resolution^2)
        value = tf.reshape(value, shape=(batch_size, num_channels, self.num_heads, head_res_dim))  # (batch_size, num_channels, num_heads, head_res_dim)
        value = tf.transpose(value, perm=[0, 2, 1, 3])  # (batch_size, num_heads, num_channels, head_res_dim)

        attentions = self.VectScaledDotProdAttention(query, key, value)  # (batch_size, num_heads, num_channels, head_res_dim)

        attentions = tf.transpose(attentions, perm=[0, 2, 1, 3])  # (batch_size, num_channels, num_heads, head_res_dim)
        attention = tf.reshape(attentions, shape=(batch_size, num_channels, resolution * resolution))  # (batch_size, num_channels, resolution^2)

        attention = tf.transpose(attention, perm=[0, 2, 1])  # (batch_size, resolution^2, num_channels)
        attention = tf.reshape(attention, shape=(batch_size, resolution, resolution, num_channels))  # (batch_size, resolution, resolution, num_channels)

        out = self.sum([x, attention])  # (batch_size, resolution, resolution, num_channels)

        return out


class LHCResBlockSmall(tf.keras.Model, ABC):
    def __init__(self, filters, kernels, strides, identity, resolution, att_num_channel, num_heads, att_embed_dim, att_kernel_size, pool_size, norm_c, name):
        super(LHCResBlockSmall, self).__init__(name='LHCResBlockSmall')
        self.Identity = identity

        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-05, name=name+"_BN1")
        self.relu1 = tf.keras.layers.Activation(activation='relu', name=name+"_Relu1")
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_Padding1')
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels[0], strides=strides[0], padding='valid', activation='linear', use_bias=False, name=name+'_Conv1')

        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-05, name=name+"_BN2")
        self.relu2 = tf.keras.layers.Activation(activation='relu', name=name+"_Relu2")
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_Padding2')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels[1], strides=strides[1], padding='valid', activation='linear', use_bias=False, name=name+'_Conv2')

        if self.Identity:
            self.convId = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels[2], strides=strides[2], padding='valid', activation='linear', use_bias=False, name=name+'_ConvId')

        self.LHC_Module = LHC_Module(pool_size=pool_size,
                                     resolution=resolution,
                                     num_channels=att_num_channel,
                                     num_heads=num_heads,
                                     head_emb_dim=att_embed_dim,
                                     kernel_size=att_kernel_size,
                                     norm_c=norm_c,
                                     name=name)

        self.add = tf.keras.layers.Add()

    def call(self, x):
        if self.Identity:
            y = self.bn1(x)
            y = self.relu1(y)
            xb = y
            y = self.pad1(y)
            y = self.conv1(y)

            y = self.bn2(y)
            y = self.relu2(y)
            y = self.pad2(y)
            y = self.conv2(y)

            y2 = self.convId(xb)

            y = self.add([y, y2])

            y = self.LHC_Module(y)

            return y
        else:
            y = self.bn1(x)
            y = self.relu1(y)
            y = self.pad1(y)
            y = self.conv1(y)

            y = self.bn2(y)
            y = self.relu2(y)
            y = self.pad2(y)
            y = self.conv2(y)

            y = self.add([y, x])

            y = self.LHC_Module(y)

            return y

    def import_w(self, layers):
        for i in range(len(layers)):
            for j in range(len(layers[i].weights)):
                self.layers[i].weights[j].assign(layers[i].weights[j])


class LHCResBlockSmall0(tf.keras.Model, ABC):
    def __init__(self, input_shape, resolution, att_num_channel, num_heads, att_embed_dim, att_kernel_size, pool_size, norm_c):
        super(LHCResBlockSmall0, self).__init__(name='LHCResBlockSmall0')

        self.Input = tf.keras.layers.InputLayer(input_shape=input_shape)

        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-05, scale=False, name='Block0_BN1')
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='Block0_Padding1')
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', activation='linear', use_bias=False, name='Block0_Conv1')
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-05, name='Block0_BN2')
        self.relu1 = tf.keras.layers.Activation(activation='relu', name='Block0_Relu1')
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='Block0_Padding2')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='Block0_MaxPool1')
        self.LHC_Module = LHC_Module(pool_size=pool_size,
                                     resolution=resolution,
                                     num_channels=att_num_channel,
                                     num_heads=num_heads,
                                     head_emb_dim=att_embed_dim,
                                     kernel_size=att_kernel_size,
                                     norm_c=norm_c,
                                     name='Module_0')

    def call(self, x):
        x1 = self.Input(x)
        x1 = self.bn1(x1)
        x1 = self.pad1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu1(x1)
        x1 = self.pad2(x1)
        x1 = self.pool1(x1)

        x1 = self.LHC_Module(x1)
        return x1

    def import_w(self, layers):
        for i in range(len(layers)):
            for j in range(len(layers[i].weights)):
                self.layers[i].weights[j].assign(layers[i].weights[j])


class ResBlockSmall(tf.keras.Model, ABC):
    def __init__(self, filters, kernels, strides, identity, name):
        super(ResBlockSmall, self).__init__(name='ResBlockSmall')
        self.Identity = identity

        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-05, name=name+"_BN1")
        self.relu1 = tf.keras.layers.Activation(activation='relu', name=name+"_Relu1")
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_Padding1')
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels[0], strides=strides[0], padding='valid', activation='linear', use_bias=False, name=name+'_Conv1')

        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-05, name=name+"_BN2")
        self.relu2 = tf.keras.layers.Activation(activation='relu', name=name+"_Relu2")
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_Padding2')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels[1], strides=strides[1], padding='valid', activation='linear', use_bias=False, name=name+'_Conv2')

        if self.Identity:
            self.convId = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels[2], strides=strides[2], padding='valid', activation='linear', use_bias=False, name=name+'_ConvId')

        self.add = tf.keras.layers.Add()

    def call(self, x):
        if self.Identity:
            y = self.bn1(x)
            y = self.relu1(y)
            xb = y
            y = self.pad1(y)
            y = self.conv1(y)

            y = self.bn2(y)
            y = self.relu2(y)
            y = self.pad2(y)
            y = self.conv2(y)

            y2 = self.convId(xb)

            y = self.add([y, y2])
            return y
        else:
            y = self.bn1(x)
            y = self.relu1(y)
            y = self.pad1(y)
            y = self.conv1(y)

            y = self.bn2(y)
            y = self.relu2(y)
            y = self.pad2(y)
            y = self.conv2(y)

            y = self.add([y, x])
            return y

    def import_w(self, layers):
        for i in range(len(layers)):
            for j in range(len(layers[i].weights)):
                self.layers[i].weights[j].assign(layers[i].weights[j])


class LHC_ResNet34(tf.keras.Model, ABC):
    def __init__(self, input_shape, num_classes, att_params):
        super(LHC_ResNet34, self).__init__(name='LHC_ResNet34')

        self.Input = tf.keras.layers.InputLayer(input_shape=input_shape)

        self.conv1 = LHCResBlockSmall0(input_shape=input_shape,
                                       resolution=56,
                                       att_num_channel=64,
                                       num_heads=att_params['num_heads'][0],
                                       att_embed_dim=att_params['att_embed_dim'][0],
                                       att_kernel_size=3,
                                       pool_size=att_params['pool_size'][0],
                                       norm_c=att_params['norm_c'][0])

        self.conv2_1 = ResBlockSmall(filters=64, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=True, name='stage1_unit1')
        self.conv2_2 = ResBlockSmall(filters=64, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage1_unit2')
        self.conv2_3 = LHCResBlockSmall(filters=64,
                                        kernels=((3, 3), (3, 3), (1, 1)),
                                        strides=((1, 1), (1, 1), (1, 1)),
                                        identity=False,
                                        resolution=56,
                                        att_num_channel=64,
                                        num_heads=att_params['num_heads'][1],
                                        att_embed_dim=att_params['att_embed_dim'][1],
                                        att_kernel_size=3,
                                        pool_size=att_params['pool_size'][1],
                                        norm_c=att_params['norm_c'][1],
                                        name='stage1_unit3')

        self.conv3_1 = ResBlockSmall(filters=128, kernels=((3, 3), (3, 3), (1, 1)), strides=((2, 2), (1, 1), (2, 2)), identity=True, name='stage2_unit1')
        self.conv3_2 = ResBlockSmall(filters=128, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage2_unit2')
        self.conv3_3 = ResBlockSmall(filters=128, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage2_unit3')
        self.conv3_4 = LHCResBlockSmall(filters=128,
                                        kernels=((3, 3), (3, 3), (1, 1)),
                                        strides=((1, 1), (1, 1), (1, 1)),
                                        identity=False,
                                        resolution=28,
                                        att_num_channel=128,
                                        num_heads=att_params['num_heads'][2],
                                        att_embed_dim=att_params['att_embed_dim'][2],
                                        att_kernel_size=3,
                                        pool_size=att_params['pool_size'][2],
                                        norm_c=att_params['norm_c'][2],
                                        name='stage2_unit4')

        self.conv4_1 = ResBlockSmall(filters=256, kernels=((3, 3), (3, 3), (1, 1)), strides=((2, 2), (1, 1), (2, 2)), identity=True, name='stage3_unit1')
        self.conv4_2 = ResBlockSmall(filters=256, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage3_unit2')
        self.conv4_3 = ResBlockSmall(filters=256, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage3_unit3')
        self.conv4_4 = ResBlockSmall(filters=256, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage3_unit4')
        self.conv4_5 = ResBlockSmall(filters=256, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)), identity=False, name='stage3_unit5')
        self.conv4_6 = LHCResBlockSmall(filters=256,
                                        kernels=((3, 3), (3, 3), (1, 1)),
                                        strides=((1, 1), (1, 1), (1, 1)),
                                        identity=False,
                                        resolution=14,
                                        att_num_channel=256,
                                        num_heads=att_params['num_heads'][3],
                                        att_embed_dim=att_params['att_embed_dim'][3],
                                        att_kernel_size=3,
                                        pool_size=att_params['pool_size'][3],
                                        norm_c=att_params['norm_c'][3],
                                        name='stage3_unit6')

        self.conv5_1 = ResBlockSmall(filters=512, kernels=((3, 3), (3, 3), (1, 1)), strides=((2, 2), (1, 1), (2, 2)),  identity=True, name='stage4_unit1')
        self.conv5_2 = ResBlockSmall(filters=512, kernels=((3, 3), (3, 3), (1, 1)), strides=((1, 1), (1, 1), (1, 1)),  identity=False, name='stage4_unit2')
        self.conv5_3 = LHCResBlockSmall(filters=512,
                                        kernels=((3, 3), (3, 3), (1, 1)),
                                        strides=((1, 1), (1, 1), (1, 1)),
                                        identity=False,
                                        resolution=7,
                                        att_num_channel=512,
                                        num_heads=att_params['num_heads'][4],
                                        att_embed_dim=att_params['att_embed_dim'][4],
                                        att_kernel_size=3,
                                        pool_size=att_params['pool_size'][4],
                                        norm_c=att_params['norm_c'][4],
                                        name='stage4_unit3')

        self.bn = tf.keras.layers.BatchNormalization(epsilon=2e-05, name='bn')
        self.relu = tf.keras.layers.Activation(activation='relu', name='relu')

        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc1 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.dp1 = tf.keras.layers.Dropout(0.4)
        self.fc2 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dp2 = tf.keras.layers.Dropout(0.4)
        self.fc3 = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def import_w(self, model):
        self.conv1.import_w(model.layers[1].layers[0:8])

        self.conv2_1.import_w(model.layers[1].layers[8:18-1])
        self.conv2_2.import_w(model.layers[1].layers[18:27-1])
        self.conv2_3.import_w(model.layers[1].layers[27:36-1])

        self.conv3_1.import_w(model.layers[1].layers[36:46-1])
        self.conv3_2.import_w(model.layers[1].layers[46:55-1])
        self.conv3_3.import_w(model.layers[1].layers[55:64-1])
        self.conv3_4.import_w(model.layers[1].layers[64:73-1])

        self.conv4_1.import_w(model.layers[1].layers[73:83-1])
        self.conv4_2.import_w(model.layers[1].layers[83:92-1])
        self.conv4_3.import_w(model.layers[1].layers[92:101-1])
        self.conv4_4.import_w(model.layers[1].layers[101:110-1])
        self.conv4_5.import_w(model.layers[1].layers[110:119-1])
        self.conv4_6.import_w(model.layers[1].layers[119:128-1])

        self.conv5_1.import_w(model.layers[1].layers[128:138-1])
        self.conv5_2.import_w(model.layers[1].layers[138:147-1])
        self.conv5_3.import_w(model.layers[1].layers[147:156-1])

        for i in range(len(self.bn.weights)):
            self.bn.weights[i].assign(model.layers[1].layers[156].weights[i])

        for i in range(len(self.relu.weights)):
            self.relu.weights[i].assign(model.layers[1].layers[157].weights[i])

        for i in range(len(self.pool.weights)):
            self.pool.weights[i].assign(model.layers[2].weights[i])

        for i in range(len(self.fc1.weights)):
            self.fc1.weights[i].assign(model.layers[3].weights[i])

        for i in range(len(self.dp1.weights)):
            self.dp1.weights[i].assign(model.layers[4].weights[i])

        for i in range(len(self.fc2.weights)):
            self.fc2.weights[i].assign(model.layers[5].weights[i])

        for i in range(len(self.dp2.weights)):
            self.dp2.weights[i].assign(model.layers[6].weights[i])

        for i in range(len(self.fc3.weights)):
            self.fc3.weights[i].assign(model.layers[7].weights[i])

    def call(self, x):
        x = self.Input(x)
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x
