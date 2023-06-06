import os
from flask import Flask, jsonify, render_template, request
import openai

from code_generator_agent import CodeGenerator

app = Flask(__name__, template_folder='template')

openai.api_key = "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"  # Replace 'YOUR_API_KEY' with your actual API key

class ChatbotAgent:
    def __init__(self):
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.user_message = {"role": "user", "content": "Hello!"}
        self.code_generator = CodeGenerator()

    def generate_response(self, user_input):
        system_messages = [self.system_message]
        user_messages = [self.user_message, {"role": "user", "content": user_input}]

        if "generate code" in user_input:
            description = self.extract_description(user_input)
            code = self.code_generator.generate_code(description)
            response = code
        else:
            response = self.generate_text('', 100, system_messages, user_messages)

        return response

    def extract_description(self, user_input):
        # Extract the description from the user's input
        # ...
        pass

    def generate_text(self, prompt, max_length, system_messages, user_messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=system_messages + user_messages,
            max_tokens=max_length,
            n=1,
            stop=None,
            temperature=0.7,
            api_key=openai.api_key
        )
        text = response.choices[0].message.content
        return text.strip()

chatbot = ChatbotAgent()

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']

    # Generate response based on the user message
    response = chatbot.generate_response(user_message)

    return jsonify({'response': response})

@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    response = chatbot.generate_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)
