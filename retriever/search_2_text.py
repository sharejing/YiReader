"""
    Time: 18-8-26 下午4:00
    Author: sharejing
    Description: Get the relevant documents of query by Baidu search (contains pre-process)

"""
import urllib
from bs4 import BeautifulSoup
import requests
import thulac
import time

thu1 = thulac.thulac(seg_only=True)


def get_format_documents(query, top_k=5):
    """
    1. 给定query，返回前top_k个相关页面，并使用url2io提取页面中的正文
    2. 格式化相关文档documents: 分词、形成reader的输入
    :param query:
    :param top_k:
    :return:
    """
    # 1. 获取百度搜索第一页html
    only_url = "http://www.baidu.com.cn/s?wd=" + urllib.parse.quote(query) + "&pn=0"
    first_page_response = urllib.request.urlopen(only_url)
    first_page_html = first_page_response.read()

    # 2. 解析第一页html，得到相关url
    soup = BeautifulSoup(first_page_html, "lxml")
    tag_h3s = soup.find_all("h3")

    top_k_documents = dict()
    top_k_documents["documents"] = []

    token = "G0Z7f4T5S3i42pVJRLNgPg"
    fields = ["title", "text"]

    # 3. 提取每一个url正文
    for h3 in tag_h3s[:top_k]:
        href = h3.find("a").get("href")

        parameters = {"token": token, "url": href, "fields": fields}
        result = requests.get("http://api.url2io.com/article", params=parameters)

        try:
            document = eval(result.text)
            if "title" in document and "text" in document:
                a_document = dict()
                a_document["title"] = document["title"]
                paragraphs = document["text"].split("\n")
                while "" in paragraphs:
                    paragraphs.remove("")

                a_document["segmented_paragraphs"] = []
                for paragraph in paragraphs:
                    a_document["segmented_paragraphs"].append(thu1.cut(paragraph, text=True).split(" "))

                top_k_documents["documents"].append(a_document)
        except NameError:
            continue

    return top_k_documents


# def analyze_url(url):
#     """
#     使用url2io解析指定url，并提取文本 [需要注册得到token]
#     :param url:
#     :return:
#     """
#
#     token = "G0Z7f4T5S3i42pVJRLNgPg"
#     fields = ["title", "text"]
#     parameters = {"token": token, "url": url, "fields": fields}
#
#     result = requests.get("http://api.url2io.com/article", params=parameters)
#
#     try:
#         document = eval(result.text)
#         if "title" in document and "text" in document:
#             a_document = dict()
#             a_document["title"] = document["title"]
#             paragraphs = document["text"].split("\n")
#             while "" in paragraphs:
#                 paragraphs.remove("")
#
#             a_document["segmented_paragraphs"] = []
#             for paragraph in paragraphs:
#                 a_document["segmented_paragraphs"].append(thu1.cut(paragraph, text=True).split(" "))
#
#             return a_document
#     except NameError:
#         return


if __name__ == "__main__":
    start = time.time()
    print(get_format_documents("苏州大学怎么样？", top_k=5))
    end = time.time()
    print(end-start)





