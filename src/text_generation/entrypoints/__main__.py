import sys

from dependency_injector.wiring import Provide, inject
from src.text_generation.dependency_injection_container import DependencyInjectionContainer
from src.text_generation.entrypoints.server import RestApiServer


@inject
def main(
    server: RestApiServer = Provide[DependencyInjectionContainer.rest_api_server]
) -> None:
    server.listen()

if __name__ == '__main__':
    container = DependencyInjectionContainer()
    container.init_resources()
    container.wire(modules=[__name__])
    main()