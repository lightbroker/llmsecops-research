# LLMSecOps Research

## Overview

This repository contains code, workflows, and experiments that support the research conducted for the paper titled "Automating Generative AI Guidelines: Reducing Prompt Injection Risk with 'Shift-Left' MITRE ATLAS Mitigation Testing". The paper is available at: https://www.sans.edu/cyber-research/automating-generative-ai-guidelines-reducing-prompt-injection-risk-shift-left-mitre-atlas-mitigation-testing/

This work also supports graduate research conducted by Adam Wilson for the M.Sc., Information Security Engineering program at SANS Technology Institute.

Paper
-----
Title: Automating Generative AI Guidelines: Reducing Prompt Injection Risk with 'Shift-Left' MITRE ATLAS Mitigation Testing

Abstract
--------
Automated testing during the build stage of the AI engineering life cycle can evaluate the
effectiveness of generative AI guidelines against prompt injection attacks. This technique
provides early feedback for developers and defenders when assessing the mitigation
performance of an LLM-integrated application. This research combines prompt
engineering techniques and automated policy violation checks in the GitHub Actions
cloud-native build system to demonstrate a practical “shift-left” approach to securing
apps based on foundation models.

What you'll find in this repo
----------------------------
- Example prompt payloads and test harnesses used to evaluate prompt injection mitigations.
- GitHub Actions workflows that run automated tests during the build stage to detect policy violations.
- Scripts and tooling that demonstrate how to integrate automated checks into an AI engineering pipeline.

Usage notes
-----------
Refer to individual directories and workflow files for details on running tests and customizing checks for your environment. The code is intended to reproduce and extend the experiments described in the paper.

# ⚠️ Disclaimer and Note on Offensive Content

Some prompts and text generation responses stored in this repository may contain offensive, biased, or otherwise harmful content. This is due to the nature of the research, which involved testing potentially adversarial inputs and model outputs.

The presence of such content is strictly for research and testing purposes only. The authors and contributors of this repository disclaim any responsibility or liability for the use, interpretation, or distribution of this material.

By accessing or using this repository, you acknowledge that:
- Offensive or harmful content may be present due to the research methodology.
- The authors are not responsible for any consequences resulting from the use of this material.
- The repository is intended solely for academic, research, and testing purposes.

License and citation
--------------------
If you use this repository for research or production, please cite the accompanying paper and follow any licensing terms included with the code.