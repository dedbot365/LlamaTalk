import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate





class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        """
        Initializes a new instance of the StreamHandler class.
        
        This constructor sets up the initial state of the handler, including the message object.
        
        Parameters:
        None
        
        Returns:
        None
        """
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        """
        Handles a new token from the LLM.

        Parameters:
            token (str): The new token from the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        """
        Handles the end of an LLM response.

        Parameters:
            response (str): The final response from the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        await self.msg.send()
        self.msg = cl.Message(content="")


# Load quantized Llama 2
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q2_K.gguf",
    model_type="llama2",
    max_new_tokens=20,
)

template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
Always provide a concise answer and use the following Context:
{context}
<</SYS>>
User:
{instruction}[/INST]"""

prompt = PromptTemplate(template=template, input_variables=["context", "instruction"])


@cl.on_chat_start
async def on_chat_start():
    """
    Event handler function that is called when a chat session starts. It initializes a memory buffer, creates an LLMChain object with the specified prompt and language model, sets the LLMChain object in the user session, and sends a welcome message to the user.

    Parameters:
        None

    Returns:
        None
    """
    memory = ConversationBufferMemory(memory_key="context")
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=memory)
    cl.user_session.set("llm_chain", llm_chain)
    await cl.Message("Model initialized. How can I help you?").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Event handler function that is called when a message is received. It retrieves the LLMChain object from the user session, invokes the LLMChain's ainvoke method with the content of the message and a configuration dictionary containing a list of callbacks.

    Parameters:
        message (cl.Message): The message object received.

    Returns:
        None
    """
    llm_chain = cl.user_session.get("llm_chain")

    await llm_chain.ainvoke(
        message.content,
        config={"callbacks": [cl.AsyncLangchainCallbackHandler(), StreamHandler()]},
    )