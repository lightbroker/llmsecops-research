# conftest.py - Shared test configuration and fixtures

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import requests
from typing import Generator, Dict, Any

# ==============================================================================
# SESSION-SCOPED FIXTURES (created once per test session)
# ==============================================================================

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
def setup_test_environment():
    """Setup run before every test automatically."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup after test
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)

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