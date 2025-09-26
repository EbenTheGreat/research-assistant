from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from backend.config.config import config
from backend.config.loader import load_yaml_config
from backend.modules.prompt_builder import build_prompt_from_config

GROQ_API_KEY = config.GROQ_API_KEY
prompt_config = load_yaml_config("backend/config/prompts.yaml")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    rag_prompt = prompt_config["rag_assistant_prompt"]
    system_prompt = build_prompt_from_config(rag_prompt)

    system_message = SystemMessagePromptTemplate.from_template(
        f"{system_prompt.template}\n\nContext:\n{{context}}"
    )

    human_message = HumanMessagePromptTemplate.from_template("Question: {input}")

    prompt = ChatPromptTemplate([system_message, human_message])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain | (lambda x: x["answer"]) | StrOutputParser()








