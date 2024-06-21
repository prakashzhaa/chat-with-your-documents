memory_prompt_template = """<s>[INST]
    You are an AI Chatbot having a conversation with human. Answer all the questions asked by humans.[/INST]
    Previous Conversation: {history}
    Human: {human_input}
    AI: """

pdf_chat_prompt_template ="""<s>[INST] Answer the user questions based on the the given context. Also consider the chat histroy. [/INST]
Chat History: {history}
Context: {context}
Question : {human_input}
Answer:"""