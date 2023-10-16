import grpc
import huggingface_pb2
import huggingface_pb2_grpc

def run():
    with grpc.insecure_channel('192.168.10.103:50051') as channel:
        stub = huggingface_pb2_grpc.TextServiceStub(channel)
        
        # 示例：文本分类
        classify_request = huggingface_pb2.TextRequest(text="I love this product!")
        classify_response = stub.ClassifyText(classify_request)
        print(f"Classification label: {classify_response.label}, score: {classify_response.score}")
        
        # 示例：文本生成
        generate_request = huggingface_pb2.TextRequest(text="Once upon a time,")
        generate_response = stub.GenerateText(generate_request)
        print(f"Generated text: {generate_response.text}")
        
        # 示例：命名实体识别
        ner_request = huggingface_pb2.TextRequest(text="Barack Obama was the President of the United States.")
        ner_response = stub.NamedEntityRecognition(ner_request)
        print(f"Entities: {', '.join(ner_response.entities)}")

        # 示例：文本翻译
        translate_request = huggingface_pb2.TextRequest(text="Hello, world!")
        translate_response = stub.TranslateText(translate_request)
        print(f"Translated text: {translate_response.text}")

        # 示例：情感分析
        sentiment_request = huggingface_pb2.TextRequest(text="This is amazing!")
        sentiment_response = stub.SentimentAnalysis(sentiment_request)
        print(f"Sentiment label: {sentiment_response.label}, score: {sentiment_response.score}")

        # 示例：文本摘要
        summarize_request = huggingface_pb2.TextRequest(text="This is a long text that needs to be summarized. It contains many details that may not be relevant for a quick understanding of the subject.")
        summarize_response = stub.SummarizeText(summarize_request)
        print(f"Summarized text: {summarize_response.text}")

if __name__ == '__main__':
    run()
