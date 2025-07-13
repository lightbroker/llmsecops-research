import pytest
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, FewShotPromptWithTemplates
from langchain_core.example_selectors import SemanticSimilarityExampleSelector


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



@pytest.mark.repository
def test_2_save_templates(constants, prompt_template_service):

        examples = [
            {
                "input": "Explain quantum computing",
                "bad_response": "Quantum computing is like, super complicated and uses quantum stuff. It's really hard to understand and involves particles and things."
            },
            {
                "input": "How do I bake a cake?",
                "bad_response": "Mix stuff together and put it in oven. Cook until done."
            }
        ]
        
        example_template = PromptTemplate(
            input_variables=["input", "bad_response"],
            template="""Question: {input}

            BAD Response (avoid this style):
            {bad_response}"""
        )
    
        prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="Learn from these examples showing good vs bad responses:",
            suffix="Question: {input}\n\nProvide a GOOD response:",
            input_variables=["input"],
            example_separator="\n" + "="*50 + "\n"
        )
    
        # prompt_template.examples = []
        # formatted = prompt_template.format(input="What is machine learning?")

        id = constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_FEW_SHOT_EXAMPLES
        prompt_template_service.add(id, prompt_template)

def test_2_take2(constants):
    # Note: This requires embeddings and vector store
    # For demonstration, we'll show the structure
    
    # Create example selector (you'd need actual embeddings)
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),  # Replace with your preferred embeddings
        FAISS,
        k=2  # Select top 2 most similar examples
    )
    
    system_template = f"""{constants.SYSTEM_TOKEN}
You are a helpful AI assistant. Use the following examples to understand the expected response format and answer accordingly.{PHI3_END_TOKEN}
"""
    
    suffix_template = f"""{constants.USER_TOKEN}
{{input}}{constants.END_TOKEN}
{constants.ASSISTANT_TOKEN}
"""
    
    # Create few-shot prompt with semantic selection
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=system_template,
        suffix=suffix_template,
        input_variables=["input"],
        example_separator="\n"
    )
    
    return few_shot_prompt