from concurrent import futures
import grpc
import huggingface_pb2
import huggingface_pb2_grpc
from transformers import pipeline

class HuggingFaceImpl(huggingface_pb2_grpc.TextServiceServicer):
    def __init__(self):
        self.classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer="distilbert-base-uncased")
        self.generator = pipeline('text-generation', model="gpt2", tokenizer="gpt2")
        self.ner = pipeline('ner', model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
        # self.translator = pipeline('translation_en_to_fr', model="t2t-base-en-fr", tokenizer="t2t-base-en-fr")
        self.translator = pipeline('translation_en_to_fr', model="Helsinki-NLP/opus-mt-en-fr", tokenizer="Helsinki-NLP/opus-mt-en-fr")
        # self.translator = pipeline('translation_en_to_fr')
        # self.sentiment = pipeline('sentiment-analysis')
        # self.summarizer = pipeline('summarization')
        # from transformers import pipeline

        # 使用特定模型进行情感分析
        self.sentiment = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment", tokenizer="nlptown/bert-base-multilingual-uncased-sentiment")

        # 使用特定模型进行文本摘要
        self.summarizer = pipeline('summarization', model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        print("init finished")

    def ClassifyText(self, request, context):
        result = self.classifier(request.text)
        return huggingface_pb2.Classification(label=result[0]['label'], score=result[0]['score'])

    def GenerateText(self, request, context):
        result = self.generator(request.text)
        return huggingface_pb2.Text(text=result[0]['generated_text'])
        
    def NamedEntityRecognition(self, request, context):
        result = self.ner(request.text)
        entities = [entity['entity'] for entity in result]
        return huggingface_pb2.Entities(entities=entities)
        
    def TranslateText(self, request, context):
        result = self.translator(request.text)
        return huggingface_pb2.Text(text=result[0]['translation_text'])
        
    def SentimentAnalysis(self, request, context):
        result = self.sentiment(request.text)
        return huggingface_pb2.Classification(label=result[0]['label'], score=result[0]['score'])
        
    def SummarizeText(self, request, context):
        result = self.summarizer(request.text)
        return huggingface_pb2.Text(text=result[0]['summary_text'])

# 创建并启动服务器
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
huggingface_pb2_grpc.add_TextServiceServicer_to_server(HuggingFaceImpl(), server)
server.add_insecure_port('[::]:50051')
server.start()
server.wait_for_termination()
