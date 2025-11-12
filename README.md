# LLMSecOps Research

## Overview

This repository contains code, workflows, and experiments that support graduate research conducted by Adam Wilson for the M.Sc., Information Security Engineering program at SANS Technology Institute. The full paper, "Automating Generative AI Guidelines: Reducing Prompt Injection Risk with 'Shift-Left' MITRE ATLAS Mitigation Testing," is available at [sans.org](https://www.sans.org/white-papers/automating-generative-ai-guidelines-reducing-prompt-injection-risk-shift-left-mitre-atlas-mitigation-testing) and [sans.edu](https://www.sans.edu/cyber-research/automating-generative-ai-guidelines-reducing-prompt-injection-risk-shift-left-mitre-atlas-mitigation-testing/).

Research Paper Abstract
--------
Automated testing during the build stage of the AI engineering life cycle can evaluate the effectiveness of generative AI guidelines against prompt injection attacks. This technique provides early feedback for developers and defenders when assessing the mitigation performance of an LLM-integrated application. This research combines prompt engineering techniques and automated policy violation checks in the GitHub Actions cloud-native build system to demonstrate a practical “shift-left” approach to securing apps based on foundation models.

Repository Contents
----------------------------
- Example prompt payloads and test harnesses used to evaluate prompt injection mitigations.
- GitHub Actions workflows that run automated tests during the build stage to detect policy violations.
- Scripts and tooling that demonstrate how to integrate automated checks into an AI engineering pipeline.
- Generative AI guidelines testing results were originally generated with GitHub Actions in [this branch](https://github.com/lightbroker/llmsecops-research/tree/scheduled-test-runs).

Usage Notes
-----------
Refer to individual directories and workflow files for details on running tests and customizing checks for your environment. The code is intended to reproduce and extend the experiments described in the paper.

License and Citation
--------------------
If you use this repository for research or production, please cite the accompanying paper and follow any licensing terms included with the code.

Disclaimer
--------------------
Read the [disclaimer](https://github.com/lightbroker/llmsecops-research/blob/main/DISCLAIMER.md).
