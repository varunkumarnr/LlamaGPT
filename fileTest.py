from typing import List, Union, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from PyPDF2 import PdfReader
import streamlit as st

# """
# Steps in Langchian data processing
# 1. Load,
# 2. Transform,
# 3. Embed,
# 4. Store,
# 5. Retrieve
# """


PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}
```
\n\n
Question: [][][][]{question}[][][][]
\n
Answer:"""


def init_page() -> None:
    st.set_page_config(page_title="Personal ChatGPT")
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI QA assistant. "
                    "When answering questions, use the context enclosed by triple backquotes if it is relevant. "
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    "Reply your answer in mardkown format."
                )
            )
        ]
        st.session_state.costs = []


def get_pdf_text() -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.

    Load and Transform
    """
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        label="Here, upload your PDF file you want ChatGPT to use to answer", type="pdf"
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
        return text_splitter.split_text(text)
    else:
        return None


def build_vectore_store(
    texts: str, embeddings: Union[LlamaCppEmbeddings, LlamaCppEmbeddings]
) -> Optional[Qdrant]:
    """
    Store the embedding vectors of text chunks into vector store (Qdrant).

    ‘embed’ and ‘store’)
    """
    if texts:
        with st.spinner("Loading PDF ..."):
            qdrant = Qdrant.from_texts(
                texts,
                embeddings,
                path=":memory:",
                collection_name="my_collection",
                force_recreate=True,
            )
        st.success("File Loaded Successfully!!")
    else:
        qdrant = None
    return qdrant


def select_llm():
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )
    return "llama-2-7b-chat.ggufv3.q2_K", temperature


def load_llm(model_name: str, temperature: float):
    """
    Load LLM.
    """
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return LlamaCpp(
        model_path="models/llama-2-7b-chat.ggufv3.q2_K.bin",
        input={"temperature": temperature, "max_length": 2048, "top_p": 1},
        n_ctx=2048,
        callback_manager=callback_manager,
        verbose=False,  # True
    )


def load_embeddings(model_name: str):
    """
    Load embedding model.
    """
    return LlamaCppEmbeddings(model_path="models/llama-2-7b-chat.ggufv3.q2_K.bin")


def get_answer(llm, messages) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """
    return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
) -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [
        {"role": find_role(message), "content": message.content} for message in messages
    ]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant
    format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def extract_userquesion_part_only(content):
    """
    Function to extract only the user question part from the entire question
    content combining user question and pdf context.
    """
    content_split = content.split("[][][][]")
    if len(content_split) == 3:
        return content_split[1]
    return content


def main():
    init_page()

    model_name, temperature = select_llm()
    llm = load_llm(model_name, temperature)
    embeddings = load_embeddings(model_name)

    texts = get_pdf_text()
    qdrant = build_vectore_store(texts, embeddings)

    init_messages()

    st.header("Personal ChatGPT")
    # Supervise user input
    # retrieve
    if user_input := st.chat_input("Input your question!"):
        if qdrant:
            context = [
                c.page_content for c in qdrant.similarity_search(user_input, k=10)
            ]
            user_input_w_context = PromptTemplate(
                template=PROMPT_TEMPLATE, input_variables=["context", "question"]
            ).format(context=context, question=user_input)
        else:
            user_input_w_context = user_input
        st.session_state.messages.append(HumanMessage(content=user_input_w_context))
        with st.spinner("Llama is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(extract_userquesion_part_only(message.content))

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


# streamlit run app.py
if __name__ == "__main__":
    main()
