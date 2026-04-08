from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import streamlit as st



from huggingface_hub import login
login()


device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "Salesforce/blip2-flan-t5-xl"
processor = Blip2Processor.from_pretrained(model_path, use_auth_token=True)
model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
        ).to(device)


def answer_question(image, question):
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True) 




#============================== STREAMLIT APP =============================


st.title("🧠 Multimodal Visual QA")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
question = st.text_input("Ask a question about the image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if question:
        answer = answer_question(image, question)
        st.write("## Answer:")
        st.write(answer)
