from dependency_injector import containers, providers

from src.text_generation.adapters.embedding_model import EmbeddingModel
from src.text_generation.adapters.text_generation_foundation_model import TextGenerationFoundationModel
from src.text_generation.entrypoints.http_api_controller import HttpApiController
from src.text_generation.entrypoints.server import RestApiServer
from src.text_generation.services.language_models.text_generation_response_service import TextGenerationResponseService
from src.text_generation.services.language_models.retrieval_augmented_generation_response_service import RetrievalAugmentedGenerationResponseService
from src.text_generation.services.similarity_scoring.generated_text_guardrail_service import GeneratedTextGuardrailService
from src.text_generation.services.logging.file_logging_service import FileLoggingService


class DependencyInjectionContainer(containers.DeclarativeContainer):

    wiring_config = containers.WiringConfiguration(modules=['src.text_generation'])
    config = providers.Configuration(yaml_files=['config.yml'])

    logging_service = providers.Singleton(
        FileLoggingService,
        filename='test.log'
    )

    foundation_model = providers.Singleton(
        TextGenerationFoundationModel
    )

    embedding_model = providers.Singleton(
        EmbeddingModel
    )
    
    rag_response_service = providers.Factory(
        RetrievalAugmentedGenerationResponseService,
        embedding_model=embedding_model
    )

    # add / implement guidelines svc
    
    # TODO placeholder; replace with responses from garak runs
    comparison_texts = [
        'something here',
        'another example'
    ]

    generated_text_guardrail_service = providers.Factory(
        GeneratedTextGuardrailService,
        embedding_model=embedding_model,
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
        logging_service=logging_service,
        listening_port=9999, # config.server.port,
        api_controller=api_controller
    )