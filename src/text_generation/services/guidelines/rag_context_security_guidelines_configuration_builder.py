from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
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
        prompt_injection_example_repository: AbstractPromptInjectionExampleRepository
    ):
        self.constants = Constants()
        self.embedding_model = embedding_model
        self.prompt_template_service = prompt_template_service
        self.prompt_injection_example_repository = prompt_injection_example_repository
        self.vectorstore = self._init_vectorstore()

    def _init_vectorstore(self):
        documents = self._load_examples()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ",", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        return FAISS.from_documents(split_docs, self.embedding_model.embeddings)

    def _load_examples(self):
        data = self.prompt_injection_example_repository.get_all()
        documents = []
        for item in data:
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

    def _create_context(self, user_prompt: str, top_k: int = 3) -> str:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        context_docs = self.vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(user_prompt)
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Example {i}:\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def get_prompt_template(self, template_id: str, user_prompt: str):
        """Get the base template from the template service and fill in RAG context"""
        # Get the base template from the template service
        base_template = self.prompt_template_service.get(id=template_id)
        
        # Get RAG context
        context = self._create_context(user_prompt)
        
        # Create a new template with the context filled in
        filled_template = PromptTemplate(
            input_variables=[self.constants.INPUT_VARIABLE_TOKEN],
            template=base_template.template.replace("{context}", context)
        )
        return filled_template

    def get_formatted_prompt(self, template_id: str, user_prompt: str) -> str:
        """Get formatted prompt with RAG context"""
        prompt_template = self.get_prompt_template(template_id, user_prompt)
        return prompt_template.format(**{self.constants.INPUT_VARIABLE_TOKEN: user_prompt})