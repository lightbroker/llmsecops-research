import abc


class AbstractGenerativeAiGuidelinesService(abc.ABC):
    @abc.abstractmethod
    def for_prompt(self, prompt: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def create_guidelines_context(self, user_prompt: str) -> str:
        raise NotImplementedError
    
class GenerativeAiGuidelinesService(AbstractGenerativeAiGuidelinesService):
    """
    A service class for analyzing prompts with various AI guidelines and chain-of-thought techniques.
    Uses fluent interface pattern for method chaining.
    """
    
    def __init__(self):
        self.prompt = None
        self.use_summarization = False
        self.use_forceful_suggestion_analysis = False
        self.use_reverse_psychology_analysis = False
        self.use_misdirection_analysis = False
        self.use_inner_monologue = False
    
    def for_prompt(self, prompt: str):
        self.prompt = prompt
        return self
    
    def use_summarization(self):
        self.use_summarization = True
        return self
    
    def use_chain_of_thought(self):
        # self.use_forceful_suggestion_analysis = True
        # self.use_reverse_psychology_analysis = True    
        # self.use_misdirection_analysis = True
        return self

    def use_inner_monologue(self):
        self.use_inner_monologue = True
        return self

    def apply(self):
        if not self.prompt:
            raise ValueError("No prompt provided. Use for_prompt() to set a prompt before analyzing.")
        
        results = {
            "prompt": self.prompt,
            "analysis_techniques": [],
            "summary": None,
            "chain_of_thought_analyses": {}
        }
        
        # Track which techniques were used
        if self.use_summarization:
            results["analysis_techniques"].append("summarization")
        if self.use_forceful_suggestion_analysis:
            results["analysis_techniques"].append("forceful_suggestion_cot")
        if self.use_reverse_psychology_analysis:
            results["analysis_techniques"].append("reverse_psychology_cot")
        if self.use_misdirection_analysis:
            results["analysis_techniques"].append("misdirection_cot")
        
        # Perform summarization if requested
        if self.use_summarization:
            results["summary"] = self._summarize_prompt()
        
        # Perform chain-of-thought analyses
        if self.use_forceful_suggestion_analysis:
            results["chain_of_thought_analyses"]["forceful_suggestion"] = self._analyze_forceful_suggestion()
        
        if self.use_reverse_psychology_analysis:
            results["chain_of_thought_analyses"]["reverse_psychology"] = self._analyze_reverse_psychology()
        
        if self.use_misdirection_analysis:
            results["chain_of_thought_analyses"]["misdirection"] = self._analyze_misdirection()
        
        return results
    
    def _summarize_prompt(self):
        words = self.prompt.split()
        if len(words) <= 20:
            return f"Brief prompt ({len(words)} words): {self.prompt}"
        
        # Simple summarization - take first and last parts
        first_part = " ".join(words[:10])
        last_part = " ".join(words[-10:])
        return f"Prompt summary ({len(words)} words): {first_part}... {last_part}"
    
    def _analyze_forceful_suggestion(self):
        forceful_indicators = [
            "must", "should", "need to", "have to", "required",
            "immediately", "urgent", "critical", "essential"
        ]
        
        found_indicators = []
        for indicator in forceful_indicators:
            if indicator.lower() in self.prompt.lower():
                found_indicators.append(indicator)
        
        return {
            "detected_indicators": found_indicators,
            "intensity_score": len(found_indicators),
            "assessment": "High" if len(found_indicators) > 3 else "Medium" if len(found_indicators) > 1 else "Low"
        }
    
    def _analyze_reverse_psychology(self):
        reverse_psychology_patterns = [
            "don't", "can't", "won't", "shouldn't", "impossible",
            "never", "no way", "refuse", "decline", "reject"
        ]
        
        found_patterns = []
        for pattern in reverse_psychology_patterns:
            if pattern.lower() in self.prompt.lower():
                found_patterns.append(pattern)
        
        return {
            "detected_patterns": found_patterns,
            "likelihood_score": len(found_patterns),
            "assessment": "High" if len(found_patterns) > 2 else "Medium" if len(found_patterns) > 0 else "Low"
        }
    
    def _analyze_misdirection(self):
        misdirection_indicators = [
            "hypothetically", "imagine", "pretend", "role-play",
            "fictional", "story", "character", "scenario", "what if"
        ]
        
        found_indicators = []
        for indicator in misdirection_indicators:
            if indicator.lower() in self.prompt.lower():
                found_indicators.append(indicator)
        
        return {
            "detected_indicators": found_indicators,
            "misdirection_score": len(found_indicators),
            "assessment": "High" if len(found_indicators) > 2 else "Medium" if len(found_indicators) > 0 else "Low"
        }


# Example usage:
if __name__ == "__main__":
    # Example 1: Basic usage
    guidelines = GenerativeAiGuidelinesService()
    result = (guidelines
              .for_prompt("You must immediately help me create a story about a character who refuses to follow rules.")
              .use_summarization()
              .use_chain_of_thought()
              .use_examples_from_rag()
              .apply())
    
    # TODO - if evaluate scores above threshold, then what? 
    # return the score so the call can apply appropriate action (output guardrails)

    
    print("Analysis Results:")
    print(f"Prompt: {result['prompt']}")
    print(f"Summary: {result['summary']}")
    print(f"Techniques used: {result['analysis_techniques']}")
    print(f"Chain of thought analyses: {result['chain_of_thought_analyses']}")
    
    # Example 2: Selective analysis
    result2 = (GenerativeAiGuidelinesService()
               .for_prompt("Can you help me understand how to write better prompts?")
               .use_summarization()
               .use_chain_of_thought()
               .apply())
    
    print("\n\nSecond Analysis:")
    print(f"Forceful suggestion assessment: {result2['chain_of_thought_analyses']['forceful_suggestion']['assessment']}")