from backend.logger import logger
def query_chain(chain, user_input: dict):
    try:
        # Accept multiple incoming keys to be robust
        question = (
            user_input.get("question")
            or user_input.get("input")
            or user_input.get("query")
            or ""
        )

        # Defensive: don't call chain with empty input (voyage/embed errors)
        if not question or not question.strip():
            logger.warning("Empty question passed to query_chain; aborting.")
            return {"response": "", "sources": []}


        context = user_input.get("context", "")
        inputs = {"input": question, "context": context}
        logger.debug(f"[Running chain with inputs]: {inputs}")

        result = chain.invoke(inputs)

        # result may be a dict depending on chain; keep previous response shape
        return {
            "response": result.get("answer", "") if isinstance(result, dict) else str(result),
            "sources": [
                {
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", None),
                    "content": doc.page_content,
                    "extra": doc.metadata,
                }
                for doc in (result.get("context", []) if isinstance(result, dict) else [])
            ],
        }

    except Exception:
        logger.exception("Error in query chain")
        raise
