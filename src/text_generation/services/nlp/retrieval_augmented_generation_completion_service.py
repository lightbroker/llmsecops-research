from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from src.text_generation.ports.abstract_embedding_model import AbstractEmbeddingModel
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.rag_guidelines_service import RetrievalAugmentedGenerationGuidelinesService
from src.text_generation.services.nlp.abstract_text_generation_completion_service import AbstractTextGenerationCompletionService
from src.text_generation.services.guidelines.abstract_rag_guidelines_service import AbstractRetrievalAugmentedGenerationGuidelinesService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService
from src.text_generation.services.utilities.response_processing_service import ResponseProcessingService


class RetrievalAugmentedGenerationCompletionService(
        AbstractTextGenerationCompletionService):
    def __init__(
            self, 
            foundation_model: AbstractFoundationModel,
            embedding_model: AbstractEmbeddingModel,
            rag_guidelines_service: AbstractRetrievalAugmentedGenerationGuidelinesService,
            response_processing_service: AbstractResponseProcessingService
        ):
        super().__init__()
        self.language_model_pipeline = foundation_model.create_pipeline()
        self.embeddings = embedding_model.embeddings
        self.rag_guidelines_service: RetrievalAugmentedGenerationGuidelinesService = rag_guidelines_service
        self.response_processing_service: ResponseProcessingService = response_processing_service

    
    def invoke(self, user_prompt: str) -> str:
        if not user_prompt:
            raise ValueError(f"Parameter 'user_prompt' cannot be empty or None")

        prompt = PromptTemplate(
            template=self.rag_guidelines_service.get_prompt_template(),
            input_variables=["context", "question"]
        )
        context = self.rag_guidelines_service.create_guidelines_context(user_prompt)
        chain = prompt | self.language_model_pipeline | StrOutputParser()
        raw_response = chain.invoke({
            "context": context,
            "question": user_prompt
        })
        response = self.response_processing_service.process_text_generation_output(raw_response)
        return response