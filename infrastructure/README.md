# Infrastructure

This directory exists to contain the foundation model (pre-trained generative language model). The foundation model files are ignored by `git`.

## Model Choice

The foundation model for this project needed to work under multiple constraints:

1. __Repo storage limits:__ Even with Git LFS enabled, GitHub restricts repository size to 5GB (at least for the free tier).
1. __Build system storage limits:__ [Standard Linux runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners?ref=devtron.ai#standard-github-hosted-runners-for-public-repositories) in GitHub Actions have a 16GB SSD.

The CPU-optimized [`microsoft/Phi-3-mini-4k-instruct-onnx`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) model met this storage space requirement. 

## Provisioning the Foundation Model

The foundation model dependency is loaded differently for local development vs. the build system:

1. __Local:__ The model is downloaded once by the `./run.sh` shell script at the project root, but excluded in `.gitignore` since it's too large for GitHub's LFS limitations.
1. __Build System:__ The model is downloaded on every workflow run with `huggingface-cli`.