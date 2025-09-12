from backend.logger import logger

def query_chain(chain, user_input: dict):
    try:
        # Ensure consistent keys
        inputs = {
            "question": user_input.get("question", ""),
            "context": user_input.get("context", ""),
        }

        logger.debug(f"[Running chain with inputs]: {inputs}")

        result = chain.invoke(inputs)

        return {
            "response": result.get("answer", ""),
            "sources": [
                {
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", None),
                    "content": doc.page_content,
                    "extra": doc.metadata,
                }
                for doc in result.get("context", [])
            ],
        }

    except Exception:
        logger.exception("Error in query chain")
        raise
