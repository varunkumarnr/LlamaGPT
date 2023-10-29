from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# template = """Question: {question}

# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.ggufv3.q2_K.bin",
    temperature=0.75,
    max_tokens=256,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
llm(prompt)


# inforamtion = """
#         Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate, investor and conspiracy theorist.[5] Musk is the founder, chairman, CEO and chief technology officer of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$207 billion as of October 2023, according to the Bloomberg Billionaires Index, and $231 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[6][7]

# Musk was born in Pretoria, South Africa, and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University in Kingston, Ontario. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics there. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and with $12 million of the money he made, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
# """


# if __name__ == "__main__":
#     print("hello")
#     summary_template = """
#             given the inforamtion {inforamtion} about a person:
#             1. a short summary
#             2. two intersting facts
#     """

#     summary_propt_template = PromptTemplate(
#         input_variables=["inforamtion"], template=summary_template
#     )

#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

#     chain = LLMChain(llm=llm, prompt=summary_propt_template)

#     print(chain.run(inforamtion=inforamtion))
