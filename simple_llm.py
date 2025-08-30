import gradio as gr
from langchain_ollama import OllamaLLM


parameters = {

    'MIN_NEW_TOKENS': 5,
    'MAX_NEW_TOKENS': 256,
    'TEMPERATURE': 0.5
}

llm = OllamaLLM(
    model="mistral",
    temperature=parameters['TEMPERATURE'],
    max_tokens=parameters['MAX_NEW_TOKENS'],
    min_tokens=parameters['MIN_NEW_TOKENS'],
)

#query = input("Please enter your query: ")

#response = llm.chat(model="mistral", messages=[{"role": "user", "content": query}], **parameters)
#response = llm.invoke(query)

#print("Response:", response)

def generate_response(prompt_txt):
    generated_response = llm.invoke(prompt_txt)
    return generated_response


# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Local Ollama AI - Sharun",
    description="Ask any question and the chatbot will try to answer."
)

chat_application.launch(server_name="127.0.0.1", server_port= 7860)