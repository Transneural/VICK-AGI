from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,GPT2LMHeadModel, GPT2Tokenizer
from web_search import WebSearcher


class DataCollector:
    def __init__(self):
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.web_searcher = WebSearcher()  # Create an instance of WebSearcher
        self.topics = ["python", "medicine", "mathematics"]  # replace with the topics you're interested in


    def generate_data_from_models(self):
        for topic in self.topics:
            # Generate text with GPT-2
            input_context = f"The past year in {topic}"
            input_ids = self.gpt_tokenizer.encode(input_context, return_tensors='pt')
            output = self.gpt_model.generate(input_ids, max_length=1000, temperature=0.7, num_return_sequences=1)
            generated_text = self.gpt_tokenizer.decode(output[0], skip_special_tokens=True)

            # Save the generated text to a file
            with open(f"{topic}.txt", "a") as f:
                f.write(generated_text + "\n")

    def scrape_data_from_web(self):
        best_match_urls = self.web_searcher.search_and_train(self.topics)
        
        for topic, url in best_match_urls.items():
            if url is not None:
                text = self.web_searcher.download_data([url])[0]

                # Save the extracted text to a file
                with open(f"{topic}.txt", "a") as f:
                    f.write(text + "\n")