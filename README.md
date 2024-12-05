<p align="center">
    <h1 align="center">Augmenta</h1>
    <em>Augment your workflows with RAG. And make it easy.</em>
</p>

## Overview

Augmenta is a powerful and flexible framework that is undergoing rapid change! Check out the flash folder for using Augmenta as a component in an interactive Flashcard app.

Augmenta provides a set of minimal yet useful abstractions that make it easy to scale your chains in complexity and add custom steps to any workflow. For a quick chatbot, a question-answering system for your documents, or scaled up applications that require retrieval and generation capabilities, Augmenta has you covered.

Need structured data from 1000+ documents? No problem. Enforcing certain fields and between-step validation functions? You got it. Build complex pipelines using Augmenta components to melt all the friction in your workflow!

## Getting Started

Get started with Augmenta in <3 minutes:

1. Clone the repository: `git clone https://github.com/TanGentleman/Augmenta.git`
2. Install dependencies: `pip install -e .`
3. Create a `.env` file with your API keys
4. Run Augmenta: `cd src && python3 -m augmenta.chat`

That's it! You're now ready to start exploring Augmenta's features and capabilities.

#### From Source (Detailed)
Prequisites:
1. Python3: This is a pure Python project, so you need to have Python installed on your system. If you don't have Python installed, you can install it your preferred way. For MacOS, I recommend using using Homebrew.
  - Install Homebrew for MacOS: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  - Install Python3: `brew install python`
  - Install git: `brew install git`

1. Clone the Augmenta repository:
```bash
git clone https://github.com/TanGentleman/Augmenta.git
```
2. Enter the project directory (venv activation for Linux/MacOS):
```bash
cd Augmenta
python3 -m venv .venv
source .venv/bin/activate
```
3. If on Windows, activate the venv using:
```bash
.venv\Scripts\activate.bat
```
4. Install the dependencies:
```bash
pip install -e .
```

### Usage

#### Setting up the .env file

Create a new file named `.env` in the root of the project directory. Use example-env.env or add the following contents:
```txt
OPENAI_API_KEY=""
TOGETHER_API_KEY=""
ANTHROPIC_API_KEY=""
LANGCHAIN_API_KEY=""
LANGCHAIN_TRACING_V2="false"
LANGCHAIN_PROJECT="Augmenta"
```
No API keys are required, as the project can run locally using LMStudio, Ollama, or Llama-cpp as a backend inference server.

## Contributing

Contributions are more than welcome! The best way to do so is creating an issue with the steps to reproduce the issue.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
