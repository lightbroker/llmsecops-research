# Change Log

### May 20, 2025

Tried multiple iterations on HTTP API and server:
1. Basic WSGI server with route handler (from *Python Cookbook*, 3rd Edition, by David Beazley and Brian K. Jones (O'Reilly)).
1. Flask API implementation
1. FastAPI implementation

The original WSGI server seemed to be the most performant, with [this run](https://github.com/lightbroker/llmsecops-research/actions/runs/14813946579) producing a successful garak test run against the Phi-3 model.

Other implementations seem to break down during the garak testing. For example, FastAPI failed to handle the garak tests in [this workflow run](https://github.com/lightbroker/llmsecops-research/actions/runs/15144678356/job/42577367897).

Refactoring to return to the original, simply WSGI server, as seen in [this commit](https://github.com/lightbroker/llmsecops-research/blob/2cb9782a4e4e11ecffe44563c8138433a0488657/.github/workflows/llmsecops-cicd.yml).