from src.text_generation.services.guidelines.abstract_rag_enhanced_semantic_similarity_guidelines_service import AbstractRagEnhancedSemanticSimilarityGuidelinesService


class RagEnhancedSemanticSimilarityGuidelinesService(AbstractRagEnhancedSemanticSimilarityGuidelinesService):
    def analyze(self, prompt_input_text: str) -> float:

        
        # TODO - check semantic similarity score
        # TODO - retry with summarized prompt? task decomposition - result could contain original score and improved score

        raise NotImplementedError