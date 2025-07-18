from src.text_generation.services.guidelines.base_security_guidelines_service import BaseSecurityGuidelinesService

class ChainOfThoughtSecurityGuidelinesService(BaseSecurityGuidelinesService):
    """Service for zero-shot chain-of-thought security guidelines."""

    def _get_template_id(self) -> str:
        return self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT_ZERO_SHOT_CHAIN_OF_THOUGHT