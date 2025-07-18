from src.text_generation.services.guidelines.base_security_guidelines_service import BaseSecurityGuidelinesService

class RagContextSecurityGuidelinesService(BaseSecurityGuidelinesService):
    """Service for RAG context security guidelines."""

    def _get_template_id(self) -> str:
        return self.constants.PromptTemplateIds.RAG_CONTEXT_SECURITY_GUIDELINES