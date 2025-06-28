from dependency_injector import containers, providers

from src.text_generation.adapters.embedding_model import EmbeddingModel
from src.text_generation.adapters.text_generation_foundation_model import TextGenerationFoundationModel
from src.text_generation.entrypoints.http_api_controller import HttpApiController
from src.text_generation.entrypoints.server import RestApiServer
from src.text_generation.services.logging.json_web_traffic_logging_service import JSONWebTrafficLoggingService
from src.text_generation.services.nlp.semantic_similarity_service import SemanticSimilarityService
from src.text_generation.services.nlp.text_generation_response_service import TextGenerationResponseService
from src.text_generation.services.nlp.retrieval_augmented_generation_response_service import RetrievalAugmentedGenerationResponseService
from src.text_generation.services.guardrails.generated_text_guardrail_service import GeneratedTextGuardrailService
from src.text_generation.services.guidelines.rag_guidelines_service import RetrievalAugmentedGenerationGuidelinesService
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
        RetrievalAugmentedGenerationGuidelinesService,
        embedding_model=embedding_model
    )

    response_processing_service = providers.Factory(
        ResponseProcessingService
    )

    rag_response_service = providers.Factory(
        RetrievalAugmentedGenerationResponseService,
        foundation_model=foundation_model,
        embedding_model=embedding_model,
        rag_guidelines_service=rag_guidelines_service,
        response_processing_service=response_processing_service
    )

    # add / implement guidelines svc
    
    # TODO placeholder; replace with responses from garak runs
    comparison_texts = [
        'something here',
        'another example'
    ]

    semantic_similarity_service = providers.Factory(
        SemanticSimilarityService,
        embedding_model=embedding_model
    )

    generated_text_guardrail_service = providers.Factory(
        GeneratedTextGuardrailService,
        semantic_similarity_service=semantic_similarity_service,
        comparison_texts=comparison_texts
    )

    text_generation_response_service = providers.Factory(
        TextGenerationResponseService,
        foundation_model
    )

    api_controller = providers.Factory(
        HttpApiController,
        logging_service=logging_service,
        text_generation_response_service=text_generation_response_service,
        rag_response_service=rag_response_service,
        generated_text_guardrail_service=generated_text_guardrail_service
    )

    rest_api_server = providers.Factory(
        RestApiServer,
        listening_port=9999, # config.server.port,
        api_controller=api_controller
    )