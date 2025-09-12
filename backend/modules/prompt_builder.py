from langchain.prompts import PromptTemplate

def build_prompt_from_config(rag_prompt: dict) -> PromptTemplate:
    template = f"""
Role: {rag_prompt['role']}

Style or Tone:
- {'\n- '.join(rag_prompt['style_or_tone'])}

Instruction:
{rag_prompt['instruction']}

Output Constraints:
- {'\n- '.join(rag_prompt['output_constraints'])}

Output Format:
- {'\n- '.join(rag_prompt['output_format'])}

Now, using the following context, answer the user’s query:


Query: {{question}}
"""

    return PromptTemplate(
        input_variables=[],
        template=template
    )

