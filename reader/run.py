"""
    Time: 18-8-26 下午2:56
    Author: sharejing
    Description: The module prepares and runs the whole system.

"""
import json
import os
import argparse
import logging
import pickle

from reader.dataset import BRCDataset
from reader.vocab import Vocab
from reader.rc_model import RCModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tensorflow的Log输出级别，只显示Error


def parse_args():

    # 命令行参数设置

    parser = argparse.ArgumentParser('brc')
    parser.add_argument('--prepare', action='store_true', help='建立文件夹，准备词典和word embedding')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--evaluate', action='store_true', help='在开发集上评价模型')
    parser.add_argument('--predict', action='store_true', help='使用训练好的模型在测试集上预测答案')
    parser.add_argument('--gpu', type=str, default='0', help='使用GPU')

    # 文件路径设置

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'], help='训练集路径list')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/demo/devset/search.dev.json'], help='开发集路径list')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'], help='测试集路径list')

    path_settings.add_argument('--vocab_dir', default='../data/vocab/', help='词表路径')
    path_settings.add_argument('--pipeline_vocab_dir', default='data/vocab/', help='pipeline词表路径')
    path_settings.add_argument('--model_dir', default='../data/models/', help='模型路径')
    path_settings.add_argument('--pipeline_model_dir', default='data/models/', help='pipeline模型路径')
    path_settings.add_argument('--result_dir', default='../data/results/', help='输出结果路径')

    # 模型设置

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'],
                                default='BIDAF', help='choose the algorithm to use')

    model_settings.add_argument('--embed_size', type=int, default=300, help='word embedding的维度')
    model_settings.add_argument('--hidden_size', type=int, default=150, help='LSTM hidden units的大小')
    model_settings.add_argument('--max_p_num', type=int, default=5, help='每一个sample中最大passage(document)数')
    model_settings.add_argument('--max_p_len', type=int, default=500, help='每一个passage的最大长度')
    model_settings.add_argument('--max_q_len', type=int, default=60, help='每一个question的最大长度')
    model_settings.add_argument('--max_a_len', type=int, default=200, help='答案的最大长度')

    # 训练参数设置

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam', help='optimizer类型')
    train_settings.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    train_settings.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1, help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32, help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10, help='train epochs')
    train_settings.add_argument('--open_batch_size', type=int, default=1, help='open test')

    return parser.parse_args()


def prepare(args):

    # 建立文件夹，准备词典和word embedding

    """
    :param args: 命令行参数
    :return:
    """

    logger = logging.getLogger("brc")

    logger.info('------------------------------------------------')
    logger.info('[1] 检查训练集|开发集|测试集是否存在......')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} 文件不存在'.format(data_path)

    logger.info('[2] 生成词表|模型|结果等文件夹......')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('[3] 预处理训练集|开发集|测试集......')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)

    """
        预处理数据集后train和dev,test集都多了question_tokens和passages字段，但passages处理方式不一样
        1) 训练集是计算answer与paragraph的最大重叠度，这已经处理好了
        2) 开发集和测试集计算question与paragraph的最大重叠度，在dataset.py自己写了代码 
        for sample in brc_data.train_set:
            print(sample['passages'])
            print(sample['question_tokens'])

        for sample in brc_data.dev_set:
            print(sample['passages'])
            print(sample['question_tokens'])
    """

    logger.info('[4] 建立词表......')
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    logger.info('[5] 过滤词表中频率小的词汇......')
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('       过滤{}个词后，当前词表大小是{}'.format(filtered_num, vocab.size()))

    logger.info('[6] 建立word embedding......')
    vocab.randomly_init_embeddings(args.embed_size)

    logger.info('[7] 存储词表所有数据......')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('[8] 预处理工作结束，快开始训练模型吧......')
    logger.info('------------------------------------------------')


def train(args):

    # 开始训练模型

    logger = logging.getLogger("brc")

    logger.info('------------------------------------------------')
    logger.info('[1] 开始加载词表和数据集......')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)

    logger.info('[2] 将数据集中的文本转化为索引序列......')
    brc_data.convert_to_ids(vocab)

    logger.info('[3] 初始化模型......')
    rc_model = RCModel(vocab, args)

    logger.info('[4] 开始训练模型......')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('[5] 模型训练成功，快开始在开发集上评价一下模型吧......')
    logger.info('------------------------------------------------')


def evaluate(args):

    # 使用预训练模型评价开发集

    logger = logging.getLogger("brc")

    logger.info('------------------------------------------------')
    logger.info('[1] 开始加载词表和数据集......')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)

    logger.info('[2] 将数据集中的文本转化为索引序列......')
    brc_data.convert_to_ids(vocab)

    logger.info('[3] 加载预训练模型......')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)

    logger.info('[4] 在开发集上评价模型......')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('       开发集上损失: {}'.format(dev_loss))
    logger.info('       开发集上Bleu和Rouge-L值:'.format(dev_bleu_rouge))
    logger.info('              Bleu-1: {}'.format(dev_bleu_rouge['Bleu-1']))
    logger.info('              Bleu-2: {}'.format(dev_bleu_rouge['Bleu-2']))
    logger.info('              Bleu-3: {}'.format(dev_bleu_rouge['Bleu-3']))
    logger.info('              Bleu-4: {}'.format(dev_bleu_rouge['Bleu-4']))
    logger.info('              RougeL: {}'.format(dev_bleu_rouge['Rouge-L']))
    logger.info('       开发集预测答案结果存储在{}'.format(os.path.join(args.result_dir)))

    logger.info('[5] 在开发集上评价成功，再试试在测试集上预测答案吧......')
    logger.info('------------------------------------------------')


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # logger.info('Running with args : {}'.format(args))
    logger.info("Config:\n%s" %
                json.dumps(vars(args), indent=4, sort_keys=True))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
