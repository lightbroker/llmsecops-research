from dependency_injector import containers, providers

from src.text_generation.adapters.embedding_model import EmbeddingModel
from src.text_generation.adapters.prompt_template_repository import PromptTemplateRepository
from src.text_generation.adapters.text_generation_foundation_model import TextGenerationFoundationModel
from src.text_generation.entrypoints.http_api_controller import HttpApiController
from src.text_generation.entrypoints.server import RestApiServer
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesService
from src.text_generation.services.guidelines.chain_of_thought_security_guidelines_service import ChainOfThoughtSecurityGuidelinesService
from src.text_generation.services.guidelines.rag_context_security_guidelines_configuration_builder import RetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.rag_context_security_guidelines_service import RagContextSecurityGuidelinesService, RetrievalAugmentedGenerationContextSecurityGuidelinesService
from src.text_generation.services.guardrails.generated_text_guardrail_service import GeneratedTextGuardrailService
from src.text_generation.services.guardrails.reflexion_security_guidelines_service import ReflexionSecurityGuardrailsService
from src.text_generation.services.logging.json_web_traffic_logging_service import JSONWebTrafficLoggingService
from src.text_generation.services.nlp.prompt_template_service import PromptTemplateService
from src.text_generation.services.nlp.semantic_similarity_service import SemanticSimilarityService
from src.text_generation.services.nlp.text_generation_completion_service import TextGenerationCompletionService
from src.text_generation.services.utilities.response_processing_service import ResponseProcessingService 


class DependencyInjectionContainer(containers.DeclarativeContainer):

    wiring_config = containers.WiringConfiguration(modules=['src.text_generation'])
    config = providers.Configuration(yaml_files=['config.yml'])

    logging_service = providers.Singleton(
        JSONWebTrafficLoggingService
    )

    foundation_model = providers.Singleton(
        TextGenerationFoundationModel
    )

    embedding_model = providers.Singleton(
        EmbeddingModel
    )
    
    rag_guidelines_service = providers.Factory(
        RetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder,
        embedding_model=embedding_model
    )

    response_processing_service = providers.Factory(
        ResponseProcessingService
    )

    rag_response_service = providers.Factory(
        RetrievalAugmentedGenerationCompletionService,
        foundation_model=foundation_model,
        embedding_model=embedding_model,
        rag_guidelines_service=rag_guidelines_service,
        response_processing_service=response_processing_service
    )

    prompt_template_repository = providers.Factory(
        PromptTemplateRepository
    )

    prompt_template_service = providers.Factory(
        PromptTemplateService,
        prompt_template_repository=prompt_template_repository
    )

    semantic_similarity_service = providers.Factory(
        SemanticSimilarityService,
        embedding_model=embedding_model
    )

    generated_text_guardrail_service = providers.Factory(
        GeneratedTextGuardrailService,
        semantic_similarity_service=semantic_similarity_service
    )

    # Register security guideline services
    chain_of_thought_guidelines = providers.Factory(
        ChainOfThoughtSecurityGuidelinesService,
        foundation_model=foundation_model,
        response_processing_service=response_processing_service,
        prompt_template_service=prompt_template_service
    ).provides(AbstractSecurityGuidelinesService)

    rag_context_guidelines = providers.Factory(
        RagContextSecurityGuidelinesService,
        foundation_model=foundation_model,
        response_processing_service=response_processing_service,
        prompt_template_service=prompt_template_service
    ).provides(AbstractSecurityGuidelinesService)
    
    reflexion_guardrails = providers.Factory(
        ReflexionSecurityGuardrailsService
    )
    
    # Main service
    text_generation_completion_service = providers.Factory(
        TextGenerationCompletionService,
        foundation_model=foundation_model,
        prompt_template_service=prompt_template_service,
        chain_of_thought_guidelines=chain_of_thought_guidelines,
        rag_context_guidelines=rag_context_guidelines,
        reflexion_guardrails=reflexion_guardrails
    )

    api_controller = providers.Factory(
        HttpApiController,
        logging_service=logging_service,
        text_generation_response_service=text_generation_completion_service,
        rag_response_service=rag_response_service,
        generated_text_guardrail_service=generated_text_guardrail_service
    )

    rest_api_server = providers.Factory(
        RestApiServer,
        listening_port=9999, # config.server.port,
        api_controller=api_controller
    )