
from src.text_generation.domain.abstract_guardrails_processed_completion import AbstractGuardrailsProcessedCompletion
from src.text_generation.services.guardrails.abstract_generated_text_guardrail_service import AbstractGeneratedTextGuardrailService


class ReflexionSecurityGuardrailsService(
    AbstractGeneratedTextGuardrailService):
    """Basic implementation of reflexion security guidelines service."""
    
    def process_generated_text(self, model_generated_text: str) -> AbstractGuardrailsProcessedCompletion:
        """
        Apply basic reflexion security guidelines        
        """
    
        return ""