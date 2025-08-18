from abc import ABC, abstractmethod
from langchain.prompts import StringPromptTemplate
from src.text_generation.ports.abstract_foundation_model import AbstractFoundationModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.chain_of_thought_security_guidelines_service import ChainOfThoughtSecurityGuidelinesService
from src.text_generation.services.guidelines.rag_context_security_guidelines_service import RagContextSecurityGuidelinesService
from src.text_generation.services.guidelines.rag_plus_cot_security_guidelines_service import RagPlusCotSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService
from src.text_generation.services.utilities.abstract_llm_configuration_introspection_service import AbstractLLMConfigurationIntrospectionService
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


class AbstractGuidelinesFactory(ABC):

    @abstractmethod
    def create_cot_guidelines_service(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder: AbstractSecurityGuidelinesConfigurationBuilder
    ) -> ChainOfThoughtSecurityGuidelinesService:
        raise NotImplementedError

    @abstractmethod
    def create_rag_context_guidelines_service(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder=AbstractSecurityGuidelinesConfigurationBuilder
    ) -> RagContextSecurityGuidelinesService:
        raise NotImplementedError
    
    @abstractmethod
    def create_rag_plus_cot_context_guidelines_service(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder: AbstractSecurityGuidelinesConfigurationBuilder
    ) -> RagPlusCotSecurityGuidelinesService:
        raise NotImplementedError

class GuidelinesFactory(AbstractGuidelinesFactory):

    def create_rag_context_guidelines_service(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder=AbstractSecurityGuidelinesConfigurationBuilder
    ) -> RagContextSecurityGuidelinesService:
        return RagContextSecurityGuidelinesService(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            llm_configuration_introspection_service=llm_configuration_introspection_service,
            config_builder=config_builder
        )
    
    def create_cot_guidelines_service(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder: AbstractSecurityGuidelinesConfigurationBuilder
    ) -> ChainOfThoughtSecurityGuidelinesService:
        return ChainOfThoughtSecurityGuidelinesService(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            llm_configuration_introspection_service=llm_configuration_introspection_service,
            config_builder=config_builder
        )

    def create_rag_plus_cot_context_guidelines_service(
        self,
        foundation_model: AbstractFoundationModel,
        response_processing_service: AbstractResponseProcessingService,
        prompt_template_service: AbstractPromptTemplateService,
        llm_configuration_introspection_service: AbstractLLMConfigurationIntrospectionService,
        config_builder: AbstractSecurityGuidelinesConfigurationBuilder
    ) -> RagPlusCotSecurityGuidelinesService: 
        return RagPlusCotSecurityGuidelinesService(
            foundation_model=foundation_model,
            response_processing_service=response_processing_service,
            prompt_template_service=prompt_template_service,
            llm_configuration_introspection_service=llm_configuration_introspection_service,
            config_builder=config_builder
        )