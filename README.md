# YiReader

A open domain question answering model

environment:
1. ubuntu16.04
2. python 3.5.2
3. tensorflow-gpu==1.0.0
4. DuReader dataset

usage:

1. python run.py --prepare
2. python run.py --train --algo BIDAF --gpu 0
3. python run.py --predict

4. python pipeline.py (from retriever to reader)
