import pytest
from langchain_core.prompts import PromptTemplate


@pytest.mark.repository
def test_1_save_templates(constants, prompt_template_service):
    
    # TODO - test load

    prompt_template = PromptTemplate(
        template="""<|system|>
            Act as a conversational agent to respond to the end user's prompt. 
            <|user|>\n
            Question: {question}<|end|>\n
            <|assistant|>
            """,
        input_variables=["question"]
    )
    id = constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_BASIC
    
    prompt_template_service.add(id, prompt_template)