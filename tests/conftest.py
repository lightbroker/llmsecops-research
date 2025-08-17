# conftest.py - Shared test configuration and fixtures

import json
import time
import pytest
import os
import random
import requests
import tempfile

from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from pathlib import Path
from tenacity import retry, stop_after_delay
from typing import Any, Dict, List

from src.text_generation import config
from src.text_generation.adapters.embedding_model import EmbeddingModel
from src.text_generation.adapters.prompt_injection_example_repository import PromptInjectionExampleRepository
from src.text_generation.adapters.prompt_template_repository import PromptTemplateRepository
from src.text_generation.adapters.foundation_models.text_generation_foundation_model import TextGenerationFoundationModel
from src.text_generation.common.constants import Constants
from src.text_generation.services.guardrails.generated_text_guardrail_service import GeneratedTextGuardrailService
from src.text_generation.services.guardrails.reflexion_security_guidelines_service import ReflexionSecurityGuardrailsService
from src.text_generation.services.guidelines.chain_of_thought_security_guidelines_service import ChainOfThoughtSecurityGuidelinesService
from src.text_generation.services.guidelines.rag_context_security_guidelines_configuration_builder import RetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder
from src.text_generation.services.guidelines.rag_context_security_guidelines_service import RagContextSecurityGuidelinesService
from src.text_generation.services.guidelines.rag_plus_cot_security_guidelines_service import RagPlusCotSecurityGuidelinesService
from src.text_generation.services.nlp.prompt_template_service import PromptTemplateService
from src.text_generation.services.nlp.semantic_similarity_service import SemanticSimilarityService
from src.text_generation.services.nlp.text_generation_completion_service import TextGenerationCompletionService
from src.text_generation.services.prompt_injection.prompt_injection_example_service import PromptInjectionExampleService
from src.text_generation.services.utilities.llm_configuration_introspection_service import LLMConfigurationIntrospectionService
from src.text_generation.services.utilities.response_processing_service import ResponseProcessingService


MAX_REQUEST_SAMPLE_COUNT = 1
MAX_RESPONSE_SAMPLE_COUNT = 50

# prompt_injection_sample_file_path = './tests/integration/samples.prompt_injection_true_positive.json'
INJECTION_PROMPT_SAMPLE_FILE_PATH = './tests/security/tests/results/01_garak_no_guidelines/process_random_requests.output.100_count.json'
PROMPT_INJECTION_PROMPTS_AND_RESPONSES_SAMPLE_FILE_PATH = './tests/security/tests/results/01_garak_no_guidelines/failed_entries.json'



def pytest_deselected(items):
    """
    Called when tests are deselected.
    Prints the nodeid of each deselected test.
    """
    if not items:
        return

    config = items[0].session.config
    reporter = config.pluginmanager.getplugin("terminalreporter")

    reporter.ensure_newline()
    reporter.section("DESELECTED TESTS", sep="=", bold=True)

    for item in items:
        reporter.line(f"Deselected: {item.nodeid}", yellow=True)

    reporter.section("END DESELECTED TESTS", sep="=", bold=True)

# ==============================================================================
# SESSION-SCOPED FIXTURES (created once per test session)
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup run before every test automatically."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["PROMPT_TEMPLATES_DIR"] = "./infrastructure/prompt_templates"
    os.environ["INJECTION_DATA_DIR"] = "./tests/security/tests/results/01_garak_no_guidelines"
    os.environ["MODEL_BASE_DIR"] = "./infrastructure/foundation_model"
    os.environ["MODEL_CPU_DIR"] = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    os.environ["MODEL_DATA_FILENAME"] = "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data"
    os.environ["MODEL_DATA_FILEPATH"] = "$MODEL_BASE_DIR/$MODEL_CPU_DIR/$MODEL_DATA_FILENAME"
    
    yield
    
    # Cleanup after test
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)
    os.environ.pop("MODEL_BASE_DIR", None)
    os.environ.pop("MODEL_CPU_DIR", None)
    os.environ.pop("MODEL_DATA_FILENAME", None)
    os.environ.pop("MODEL_DATA_FILEPATH", None)

@pytest.fixture(scope="session")
def constants():
    return Constants()

@pytest.fixture(scope="session")
def embedding_model():
    return EmbeddingModel()

@pytest.fixture(scope="session")
def prompt_template_repository():
    return PromptTemplateRepository()

@pytest.fixture(scope="session")
def prompt_template_service(prompt_template_repository):
    return PromptTemplateService(prompt_template_repository)

@pytest.fixture(scope="session")
def prompt_injection_example_repository():
    return PromptInjectionExampleRepository()

@pytest.fixture(scope="session")
def rag_config_builder(
        embedding_model,
        prompt_template_service,
        prompt_injection_example_repository):
    return RetrievalAugmentedGenerationSecurityGuidelinesConfigurationBuilder(
        embedding_model=embedding_model,
        prompt_template_service=prompt_template_service,
        prompt_injection_example_repository=prompt_injection_example_repository
    )

@pytest.fixture(scope="session")
def llm_configuration_introspection_service():
    return LLMConfigurationIntrospectionService()

@pytest.fixture(scope="session")
def rag_context_guidelines(
        foundation_model,
        response_processing_service,
        prompt_template_service,
        llm_configuration_introspection_service,
        rag_config_builder):
    return RagContextSecurityGuidelinesService(
        foundation_model=foundation_model,
        response_processing_service=response_processing_service,    
        prompt_template_service=prompt_template_service,
        llm_configuration_introspection_service=llm_configuration_introspection_service,
        config_builder=rag_config_builder
    )

@pytest.fixture(scope="session")
def chain_of_thought_guidelines(
        foundation_model,
        response_processing_service,
        llm_configuration_introspection_service,
        prompt_template_service):
    return ChainOfThoughtSecurityGuidelinesService(
        foundation_model=foundation_model,
        response_processing_service=response_processing_service,
        llm_configuration_introspection_service=llm_configuration_introspection_service,
        prompt_template_service=prompt_template_service
    )

@pytest.fixture(scope="session")
def rag_plus_cot_guidelines(
        foundation_model,
        response_processing_service,
        prompt_template_service,
        llm_configuration_introspection_service,
        rag_config_builder):
    return RagPlusCotSecurityGuidelinesService(
        foundation_model=foundation_model,
        response_processing_service=response_processing_service,    
        prompt_template_service=prompt_template_service,
        llm_configuration_introspection_service=llm_configuration_introspection_service,
        config_builder=rag_config_builder
    )

@pytest.fixture(scope="session")
def prompt_injection_example_service(prompt_injection_example_repository):
    return PromptInjectionExampleService(repository=prompt_injection_example_repository)

@pytest.fixture(scope="session")
def reflexion_guardrails(
        foundation_model,
        semantic_similarity_service,
        prompt_injection_example_service,
        response_processing_service,
        prompt_template_service,
        llm_configuration_introspection_service):
    return ReflexionSecurityGuardrailsService(
        foundation_model=foundation_model,
        semantic_similarity_service=semantic_similarity_service,
        prompt_injection_example_service=prompt_injection_example_service,
        response_processing_service=response_processing_service,    
        prompt_template_service=prompt_template_service,
        llm_configuration_introspection_service=llm_configuration_introspection_service
    )

@pytest.fixture(scope="session")
def response_processing_service():
    return ResponseProcessingService()

@pytest.fixture(scope="session")
def llm_configuration_introspection_service():
    return LLMConfigurationIntrospectionService()

@pytest.fixture(scope="session")
def text_generation_completion_service(
        response_processing_service,
        prompt_template_service,
        chain_of_thought_guidelines,
        rag_context_guidelines,
        rag_plus_cot_guidelines,
        reflexion_guardrails,
        semantic_similarity_service,
        prompt_injection_example_service,
        llm_configuration_introspection_service):
    return TextGenerationCompletionService(
        response_processing_service=response_processing_service,
        prompt_template_service=prompt_template_service,
        chain_of_thought_guidelines=chain_of_thought_guidelines,
        rag_context_guidelines=rag_context_guidelines,
        rag_plus_cot_guidelines=rag_plus_cot_guidelines,
        reflexion_guardrails=reflexion_guardrails,
        semantic_similarity_service=semantic_similarity_service,
        prompt_injection_example_service=prompt_injection_example_service,
        llm_configuration_introspection_service=llm_configuration_introspection_service
    )

@pytest.fixture(scope="session")
def semantic_similarity_service(embedding_model):
    return SemanticSimilarityService(embedding_model)

@pytest.fixture(scope="session")
def guardrail_service(
        semantic_similarity_service,
        prompt_injection_sample_responses):
    return GeneratedTextGuardrailService(
        semantic_similarity_service=semantic_similarity_service, 
        comparison_texts=prompt_injection_sample_responses)

@pytest.fixture(scope="session")
def math_prompts() -> list[str]:

    # Generate 100 simple math problem prompts
    math_prompts: list[str] = []

    # Define operation templates
    templates = [
        # Addition templates
        "What is {a} + {b}?",
        "Add {a} to {b}",
        "Find the sum of {a} and {b}",
        "What do you get when you add {a} and {b}?",
        
        # Subtraction templates
        "What is {a} - {b}?",
        "Subtract {b} from {a}",
        "Find the difference between {a} and {b}",
        "What do you get when you subtract {b} from {a}?",
        
        # Multiplication templates
        "What is {a} x {b}?",
        "What is {a} * {b}?",
        "Multiply {a} by {b}",
        "Find the product of {a} and {b}",
        "What do you get when you multiply {a} and {b}?",
        
        # Division templates
        "What is {a} ÷ {b}?",
        "What is {a} / {b}?",
        "Divide {a} by {b}",
        "What do you get when you divide {a} by {b}?",
    ]

    for i in range(100):
        template = random.choice(templates)
        
        # Generate operands between 1 and 10
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        
        # For subtraction, ensure a >= b to avoid negative results
        if "subtract" in template.lower() or " - " in template:
            if a < b:
                a, b = b, a
        
        # For division, ensure a is divisible by b or use different approach
        if "÷" in template or "/" in template or "divide" in template.lower():
            # Make sure we get clean division by making a a multiple of b
            a = b * random.randint(1, 10//b if b <= 10 else 1)
            # Ensure a doesn't exceed 10, so recalculate if needed
            if a > 10:
                b = random.randint(1, 10)
                a = b * random.randint(1, 10//b)
        
        prompt = template.format(a=a, b=b)
        math_prompts.append(prompt)
    
    return math_prompts



@pytest.fixture(scope="session")
def malicious_prompts():
    json_data = load_json_file(INJECTION_PROMPT_SAMPLE_FILE_PATH)
    requests = extract_requests(json_data, use_all=True)
    return requests

@pytest.fixture(scope="session")
def prompt_injection_sample_requests():
    json_data = load_json_file(PROMPT_INJECTION_PROMPTS_AND_RESPONSES_SAMPLE_FILE_PATH)
    requests = extract_requests(json_data)
    return requests

@pytest.fixture(scope="session")
def prompt_injection_sample_responses():
    json_data = load_json_file(PROMPT_INJECTION_PROMPTS_AND_RESPONSES_SAMPLE_FILE_PATH)
    responses = extract_responses(json_data)
    return responses

@pytest.fixture(scope="session")
def prompt_injection_all_example_responses():
    json_data = load_json_file(PROMPT_INJECTION_PROMPTS_AND_RESPONSES_SAMPLE_FILE_PATH)
    responses = extract_responses(json_data, use_all=True)
    return responses

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        raise
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        raise

def extract_requests(data: List[Dict[str, Any]], use_all=False) -> List[str]:
    requests = []
    for item in data:
        if 'request' in item:
            requests.append(item['request'])
        else:
            print(f"Warning: Item missing 'request' field: {item}")
    
    if use_all:
        return requests
    
    return random.sample(requests, k=MAX_REQUEST_SAMPLE_COUNT)

def extract_responses(data: List[Dict[str, Any]], use_all=False) -> List[str]:
    responses = []
    for item in data:
        if 'response' in item:
            responses.append(item['response'])
        else:
            print(f"Warning: Item missing 'response' field: {item}")
    
    if use_all:
        return responses

    count = min(len(responses), MAX_RESPONSE_SAMPLE_COUNT)
    return random.sample(responses, k=count)

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "api_base_url": "http://localhost:8000/api",
        "timeout": 30,
        "max_retries": 3,
        "test_user_email": "test@example.com",
        "debug": True
    }

# ==============================================================================
# MODULE-SCOPED FIXTURES (created once per test module)
# ==============================================================================

@pytest.fixture(scope="module")
def api_client():
    """HTTP client for API testing."""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json"
    })
    yield session
    session.close()

# ==============================================================================
# FUNCTION-SCOPED FIXTURES (created for each test function)
# ==============================================================================

@retry(stop=stop_after_delay(10))
def wait_for_responsive_http_api():
    return requests.get(config.get_api_url())

@pytest.fixture
def restart_api():
    (Path(__file__).parent / "../src/text_generation/entrypoints/server.py").touch()
    time.sleep(0.5)
    wait_for_responsive_http_api()

@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "secure_password123",
        "first_name": "Test",
        "last_name": "User"
    }

@pytest.fixture
def sample_users():
    """Multiple sample users for testing."""
    return [
        {"username": "user1", "email": "user1@example.com"},
        {"username": "user2", "email": "user2@example.com"},
        {"username": "user3", "email": "user3@example.com"},
    ]

@pytest.fixture
def mock_user_service():
    """Mock user service for unit testing."""
    mock = Mock()
    mock.get_user.return_value = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com"
    }
    mock.create_user.return_value = {"id": 1, "success": True}
    mock.delete_user.return_value = True
    return mock

@pytest.fixture
def mock_external_api():
    """Mock external API responses."""
    mock = MagicMock()
    mock.get.return_value.json.return_value = {"status": "success", "data": []}
    mock.get.return_value.status_code = 200
    mock.post.return_value.json.return_value = {"id": 123, "created": True}
    mock.post.return_value.status_code = 201
    return mock

@pytest.fixture
def temp_directory():
    """Create temporary directory for file testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_files(temp_directory):
    """Create sample files for testing."""
    files = {}
    
    # Create text file
    text_file = temp_directory / "sample.txt"
    text_file.write_text("Hello, World!")
    files["text"] = text_file
    
    # Create JSON file
    json_file = temp_directory / "sample.json"
    json_file.write_text('{"name": "test", "value": 123}')
    files["json"] = json_file
    
    # Create CSV file
    csv_file = temp_directory / "sample.csv"
    csv_file.write_text("name,age,city\nJohn,30,NYC\nJane,25,LA")
    files["csv"] = csv_file
    
    return files

@pytest.fixture
def frozen_time():
    """Fix time for testing time-dependent code."""
    fixed_time = datetime(2024, 1, 15, 12, 0, 0)
    
    class MockDatetime:
        @classmethod
        def now(cls):
            return fixed_time
        
        @classmethod
        def utcnow(cls):
            return fixed_time
    
    # You would typically use freezegun or similar library
    # This is a simplified example
    return MockDatetime

# ==============================================================================
# PARAMETRIZED FIXTURES
# ==============================================================================

@pytest.fixture(params=[1, 5, 10, 100])
def batch_size(request):
    """Different batch sizes for testing."""
    return request.param

# ==============================================================================
# AUTOUSE FIXTURES (automatically used by all tests)
# ==============================================================================

@pytest.fixture(autouse=True)
def log_test_info(request):
    """Log test information automatically."""
    print(f"\n=== Running test: {request.node.name} ===")
    yield
    print(f"=== Finished test: {request.node.name} ===")

# ==============================================================================
# CONDITIONAL FIXTURES
# ==============================================================================

@pytest.fixture
def authenticated_user(request, sample_user_data):
    """Fixture that creates authenticated user context."""
    # Check if test is marked as requiring authentication
    if hasattr(request, 'node') and 'auth_required' in request.node.keywords:
        # Create authenticated user session
        return {
            "user": sample_user_data,
            "token": "fake-jwt-token",
            "expires": datetime.now() + timedelta(hours=1)
        }
    return None

# ==============================================================================
# PYTEST HOOKS (customize pytest behavior)
# ==============================================================================

def pytest_configure(config):
    """Configure pytest before tests run."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "auth_required: mark test as requiring authentication"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external_service: mark test as requiring external service"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location or name
    for item in items:
        # Mark all tests in integration folder as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests with 'slow' in name as slow
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark external API tests
        if "external" in item.name.lower() or "api" in item.name.lower():
            item.add_marker(pytest.mark.external_service)

def pytest_runtest_setup(item):
    """Setup before each test runs."""
    # Skip tests marked as external_service if no network
    if "external_service" in item.keywords:
        if not hasattr(item.config, 'option') or getattr(item.config.option, 'skip_external', False):
            pytest.skip("Skipping external service test")

def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test."""
    # Add any global cleanup logic here
    pass

def pytest_report_teststatus(report, config):
    """Customize test status reporting."""
    # You can customize how test results are reported
    pass

# ==============================================================================
# CUSTOM PYTEST MARKERS
# ==============================================================================

# These can be used with @pytest.mark.marker_name in tests
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

# ==============================================================================
# FIXTURE COMBINATIONS
# ==============================================================================

@pytest.fixture
def api_client_with_auth(api_client, authenticated_user):
    """API client with authentication headers."""
    if authenticated_user:
        api_client.headers.update({
            "Authorization": f"Bearer {authenticated_user['token']}"
        })
    return api_client