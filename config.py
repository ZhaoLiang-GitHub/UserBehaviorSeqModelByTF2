class Feature(object):
    def __init__(self, name="", feature_type="other", data_type="dense", length=1, progress="org", embedding_size=1,
                 category=1):
        """
        特征类，用于描述一个在数据集中一列的
        :param name: 特征名称
        :param feature_type: 特征类型，在用户序列模型中，包括了，item、sequence、other
        :param data_type: 数据类型，分为离散型、浮点型、向量型
        :param length: 每个特征的数据长度，
        :param progress: 每个特征的处理方式，直接计算和向量化
        :param embedding_size: 向量化编码长度
        :param category: 离散型特征每个特征个数
        """
        self.name = name
        self.feature_type = feature_type
        self.data_type = data_type
        self.length = length
        self.progress = progress
        self.embedding_size = embedding_size
        self.category = category


class Config(object):
    def __init__(self, *args, **kwargs):
        self.train_path = "./data/train/part*"
        self.test_path = "./data/train/part*"
        self.valid_path = "./data/train/part*"
        self.feature = [
            Feature("designer_day7_download_ratio_pv", feature_type="item"),
            Feature("designer_day7_click_ratio_pv", feature_type="item"),

            Feature("seq_designer_day7_download_ratio_pv", feature_type="sequence", length=5),
            Feature("seq_designer_day7_click_ratio_pv", feature_type="sequence", length=5),

            Feature("userVggVec", data_type="embedding", length=125),
            Feature("dayOfWeek", data_type="sparse", category=2, embedding_size=8),
            Feature("isWeekend", data_type="sparse", category=7, embedding_size=8),
            Feature("udesigner_day7_download_pv"),
            Feature("udesigner_day7_click_ratio"),
            Feature("udesigner_day7_download_ratio"),
            Feature("udesigner_day14_click_ratio"),
            Feature("udesigner_day14_download_ratio"),
            Feature("udesigner_day28_click_ratio"),
            Feature("udesigner_day28_download_ratio"),
            # Feature("vggEmbedding_sim"),
            Feature("vgg16_pca", data_type="embedding", length=125)
        ]
        self.label = "is_click"

        self.batch_size = 1024
        self.seed = 1024
        self.l2_reg_embedding = 1e-5
        self.hidden_layer_units = [256, 128, 64]
        self.activation = "relu"
        self.dropout_rate = 0.2
        self.use_bn = True
        self.output_activation = "sigmoid"

        self.ckpt_path = "./data/ckpt"
        self.log_path = "./data/log"
        self.epochs = 2
