# Security Policy

This repository holds research code, data samples, and documents. It is not a deployed service or a packaged library, and the code is meant to be run by researchers on their own machines or on Kaggle.

## Supported Versions

Only the latest commit on `main` is maintained. The program is closed, so fixes are limited to security problems and errors in the record.

## Reporting a Vulnerability

If you find a security problem, for example a script that executes untrusted input, a committed secret or token, or a dependency with a known vulnerability that affects how the code is used here, please report it privately:

- Use GitHub's **"Report a vulnerability"** button under this repository's Security tab (private vulnerability reporting), or
- contact me through my GitHub profile ([@awabmoha](https://github.com/awabmoha)).

Please do not open a public issue with the details. I will acknowledge a report within a week and say what I will do about it.

## Scope Notes

- The scripts download public models and datasets (Hugging Face, Kaggle). Treat those downloads as you would any third-party code and data.
- Trust scores produced by this code are research outputs, not safety mechanisms. Do not use them to make deployment or safety decisions.
