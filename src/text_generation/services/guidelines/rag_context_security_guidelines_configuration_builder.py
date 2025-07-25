from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.text_generation.adapters.embedding_model import EmbeddingModel
from src.text_generation.common.constants import Constants
from src.text_generation.ports.abstract_prompt_injection_example_repository import AbstractPromptInjectionExampleRepository
from src.text_generation.ports.abstract_embedding_model import AbstractEmbeddingModel
from src.text_generation.services.guidelines.abstract_security_guidelines_service import AbstractSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.nlp.abstract_prompt_template_service import AbstractPromptTemplateService


class RetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder(
        AbstractSecurityGuidelinesConfigurationBuilder):

    def __init__(
            self,
            embedding_model: AbstractEmbeddingModel,
            prompt_template_service: AbstractPromptTemplateService,
            prompt_injection_example_repository: AbstractPromptInjectionExampleRepository):
        self.constants = Constants()
        self.embedding_model: EmbeddingModel = embedding_model
        self.prompt_template_service = prompt_template_service
        self.prompt_injection_example_repository = prompt_injection_example_repository
        self.prompt_template_id = self.constants.PromptTemplateIds.PHI_3_MINI_4K_INSTRUCT__03_FEW_SHOT_EXAMPLES
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self):
        documents = self._load_examples()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ",", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create FAISS vector store from chunks
        return FAISS.from_documents(split_docs, self.embedding_model.embeddings)

    def _load_examples(self) -> list[Document]:
        data = self.prompt_injection_example_repository.get_all()
        print(f'got {len(data)} prompt injection examples')

        documents = []
        for item in data:
            # Create document content combining both fields for better retrieval
            content = f"Prompt: {item['prompt_injection_prompt']}\nCompletion: {item['prompt_injection_completion']}"
            doc = Document(
                page_content=content,
                metadata={
                    "prompt_injection_prompt": item['prompt_injection_prompt'],
                    "prompt_injection_completion": item['prompt_injection_completion']
                }
            )
            documents.append(doc)
        
        return documents
    

    def _create_few_shot_prompt(self, user_prompt: str) -> FewShotPromptTemplate:
        prompt_template: FewShotPromptTemplate = self.prompt_template_service.get(id=self.prompt_template_id)
        prompt_template.examples = self._load_examples()
        return prompt_template


    def get_prompt_template(self, user_prompt: str):
        return self._create_few_shot_prompt(user_prompt)
    