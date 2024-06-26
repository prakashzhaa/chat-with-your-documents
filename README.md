# Chat with your documents

It is an innovative project focusing on building a sophisticated multimodel AI chatbot application. This project integrates various AI models to handle audio, images, and PDFs within a single chat interface. It is an exciting venture for anyone interested in AI and software development, providing a platform to explore and implement cutting-edge technologies.


The core components of this project include WishperAI for audio processing, LLaVA for image handling, and ChromaDB for managing PDFs. These language models come together to create a cohesive and dynamic chat experience, allowing users to interact with the system in multiple ways.


However, this project is still a work in progress. There's a lot of potential for enhancement and optimization.

## Characteristics
* **Quantization Models**
 This app leverages the power of 'quantized models'. These models are particularly special because they are tailored to run efficiently on regular consumer hardware, like the computers most of us use at home or in our lab. Typically, the original versions of these models are quite large and require more powerful machines to operate. However, quantized models are optimized to be smaller and more efficient, maintaining much of their performance. This optimization means you can use this model and its features without needing a high-end computer. The quantized models used in this application are  [TheBloke-Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
* **Audio Chatting with WishperAI by OpenAI**
  Harnessing the powerful transciption capabilities of WishperAI, this model delivers a sophisticated audio messaging experience. With Wishper model integration, the app can accurately interpret and respond to voice inputs, ensuring a seamless and natural flow in conversations. The Wishper models excel at providing precise transcriptions, making your audio interactions smooth and intuitive. [Wishper-small] (https://huggingface.co/openai/whisper-small)
* **Image Chatting with LLaVA**
  The app uses the LLaVA model for image processing, a fine-tuned LLaVA model that understands image embeddings generated by a CLIP model. This integration enhances text and image comprehension, making chat interactions more engaging and interactive, especially with visual content. LLaVA acts as a sophisticated pipeline for advanced image and text integration. (The llama-cpp-python)[https://github.com/abetlen/llama-cpp-python]
* **PDF Chatting with ChromaDB**
The app integrates ChromaDB as a vector database for efficient PDF handling. This feature enables users to engage with their PDF files locally on their devices. Whether reviewing business reports, academic papers, or other documents, the app provides a seamless experience. It offers insightful interactions with PDFs, making it a valuable tool for extracting summaries and engaging in unique dialogues with the text.
## User Interface
![user-interface](https://github.com/prakashzhaa/chat-with-your-documents/assets/73091946/8ecc7243-fc43-4ae7-b022-1267da06ba3e)
## Getting Started 
To get started with you chatbot application, clone the repository and follow these steps:
1. **create a conda enviroment:**
   `conda create --name new_env python=3.9`
   `conda activate new_env`
   `conda update --name new_env --all`
3. **Install Requirements:**
    `pip install -r requirements.txt`

4. **Locally set quantized models:**
   Download all the models you want to implement.
   * Quantized Mistral Models: (small)[https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q3_K_M.gguf]
                               (large)[https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf]
   * LLaVa Model: (Clip Model)[https://huggingface.co/mys/ggml_llava-v1.5-7b/blob/main/mmproj-model-f16.gguf]
                  (LLaVA-small)[https://huggingface.co/mys/ggml_llava-v1.5-7b/blob/main/ggml-model-q4_k.gguf]
                   (LLaVA-large)[https://huggingface.co/mys/ggml_llava-v1.5-7b/blob/main/ggml-model-q5_k.gguf]
5. **Customize config file**
   Check the config file and change the path according to the models.

6. **Run application**
   * Initialize chat sessions: `python database_operations.py`
   * Run Streamlit: `streamlit run app.py`
     
## Architecture of the application

![Architecture-of-the-application](https://github.com/prakashzhaa/chat-with-your-documents/assets/73091946/43996f94-8f33-4fc2-9487-b2e6f947f7ef)

