from src.text_generation.common.constants import Constants
from src.text_generation.services.utilities.abstract_response_processing_service import AbstractResponseProcessingService


class ResponseProcessingService(AbstractResponseProcessingService):

    def __init__(self):
        self.constants = Constants()

    def process_text_generation_output(self, raw_output: str) -> str:
        if self.constants.ASSISTANT_TOKEN in raw_output:
            # split at assistant token and take everything after it
            parts = raw_output.split(self.constants.ASSISTANT_TOKEN)
            answer = parts[-1].strip()
            # remove trailing <|end|> tokens if present
            if answer.endswith(self.constants.END_TOKEN):
                answer = answer[:-(len(self.constants.END_TOKEN))].strip()
            return answer
        else:
            # return raw original (fallback)
            return raw_output.strip()