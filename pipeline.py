"""
    Time: 18-8-26 下午9:02
    Author: sharejing
    Description: YiReader. from retriever to reader

"""
import os
import pickle
from reader.run import parse_args
from reader.rc_model import RCModel
from reader.dataset import BRCDataset
from retriever.search_2_text import get_format_documents
import prettytable
import code
from retriever.search_2_text import thu1

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print("Load vocab")
with open(os.path.join(args.pipeline_vocab_dir, "vocab.data"), "rb") as fin:
    vocab = pickle.load(fin)

print("Load model")
rc_model = RCModel(vocab, args)
rc_model.restore(model_dir=args.pipeline_model_dir, model_prefix=args.algo)
print("Load succeed")


def process(query):
    """
    给定query，返回预测答案
    :param query:
    :return:
    """
    relevant_documents = get_format_documents(query)
    query_tokens = thu1.cut(query, text=True).split(" ")
    relevant_documents["question"] = query
    relevant_documents["segmented_question"] = query_tokens

    data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len)
    data.load_external_dataset(relevant_documents)
    data.convert_to_ids(vocab)
    result_batches = data.gen_mini_batches("test", args.open_batch_size,
                                           pad_id=vocab.get_id(vocab.pad_token),
                                           shuffle=False)

    predictions = rc_model.one_evaluate(result_batches)

    table = prettytable.PrettyTable(["rank", "query", "answers"])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, query, p])
    print(table)


banner = """
My communication Reader
>> process(question)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())

process("苏州大学怎么样？")
