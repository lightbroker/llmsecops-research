from itertools import product
from src.text_generation.domain.abstract_guidelines_processed_completion import AbstractGuidelinesProcessedCompletion
from src.text_generation.domain.guardrails_processed_completion import GuardrailsProcessedCompletion
from src.text_generation.services.guidelines.abstract_generative_ai_security_guidelines_service import AbstractGenerativeAiSecurityGuidelinesService
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService


class GenerativeAiSecurityGuidelinesService(
    AbstractGenerativeAiSecurityGuidelinesService):
    """
        A service class for analyzing prompts with various AI guidelines and chain-of-thought techniques.
        Uses fluent interface pattern for method chaining.
    """
    def __init__(
            self,
            prompt_template_service: AbstractPromptTemplateService):
        # services
        self.prompt_template_service = prompt_template_service
        # properties
        self.prompt = None
        self.is_chain_of_thought_enforced = False
        self.is_rag_example_usage_enforced = False

    # private methods

    def _iterate_all_combinations(self):
        """
        Iterate through all possible combinations of the two boolean properties.
        
        Yields:
            tuple: (is_chain_of_thought_enforced, is_rag_example_usage_enforced)
        """
        # Get all possible combinations of True/False for 2 boolean properties
        combinations = product([True, False], repeat=2)
        
        for cot_enforced, rag_enforced in combinations:
            # Set the properties
            self.is_chain_of_thought_enforced = cot_enforced
            self.is_rag_example_usage_enforced = rag_enforced
            
            # Yield the current combination for processing
            yield (cot_enforced, rag_enforced)

    def _process_all_enforced_guideline_techniques(self) -> AbstractGuidelinesProcessedCompletion:
        for i, (cot, rag) in enumerate(self.iterate_all_combinations(), 1):
            print(f"\n=== Combination {i}: CoT={cot}, RAG={rag} ===")
            
            if not cot and not rag:
                # Case 1: Neither chain of thought nor RAG enforced
                print("Running basic processing without enhanced reasoning or examples")
                result = self._process_basic()
                
            elif not cot and rag:
                # Case 2: Only RAG examples enforced
                print("Running with RAG examples but no chain of thought")
                result = self._process_with_rag_only()
                
            elif cot and not rag:
                # Case 3: Only chain of thought enforced
                print("Running with chain of thought but no RAG examples")
                result = self._process_with_cot_only()
                
            else:  # cot and rag
                # Case 4: Both chain of thought and RAG enforced
                print("Running with both chain of thought and RAG examples")
                result = self._process_with_cot_and_rag()
            
            # Store or analyze result
            self._store_result(result, cot, rag)
        
        # Reset to original state
        self.is_chain_of_thought_enforced = False
        self.is_rag_example_usage_enforced = False
        processed_completion = GuardrailsProcessedCompletion(
            score=0.5,
            cosine_similarity_risk_threshold=0.7,
            original_completion="test",
            final="test2"
        )
        return processed_completion

    def _process_basic(self):
        return {
            'method': 'basic',
            'steps': ['direct_inference'],
            'examples_used': 0,
            'reasoning_depth': 'shallow'
        }
    
    def _process_with_rag_only(self):
        return {
            'method': 'rag_only',
            'steps': ['retrieve_examples', 'apply_examples', 'generate_response'],
            'examples_used': 3,
            'reasoning_depth': 'shallow'
        }

    def _process_with_cot_only(self):
        return {
            'method': 'cot_only',
            'steps': ['analyze_problem', 'break_down_steps', 'reason_through', 'conclude'],
            'examples_used': 0,
            'reasoning_depth': 'deep'
        }

    def _process_with_cot_and_rag(self):
        return {
            'method': 'cot_and_rag',
            'steps': ['retrieve_examples', 'analyze_with_context', 'reason_step_by_step', 'synthesize_with_examples', 'conclude'],
            'examples_used': 5,
            'reasoning_depth': 'deep'
        }

    # end private methods

    def for_prompt(self, prompt: str):
        self.prompt = prompt
        return self

    def use_chain_of_thought(self):
        # TODO need prompt template
        # self.use_forceful_suggestion_analysis = True
        # self.use_reverse_psychology_analysis = True    
        # self.use_misdirection_analysis = True
        self.is_chain_of_thought_enforced = True
        # TODO - this is a given... self.use_summarization = True
        return self

    def use_examples_from_rag(self):
        self.is_rag_example_usage_enforced = True
        return self

    def apply(self) -> AbstractGuidelinesProcessedCompletion:
        if not self.prompt:
            raise ValueError("No prompt provided. Use `for_prompt()` to set a prompt before analyzing.")

        self._process_all_enforced_guideline_techniques()

        results = {
            "prompt": self.prompt,
            "analysis_techniques": [],
            "summary": None,
            "chain_of_thought_analyses": {}
        }

        return results
