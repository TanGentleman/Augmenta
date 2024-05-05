<p align="center">
    <h1 align="center">Augmenta</h1>
    <em>Augment your workflows with RAG. And make it easy.</em>
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
</details>
<hr>

## Overview

Augmenta is a project that aims to deconstruct RAG (Retrieval-Augmented Generation) tasks. It's flexible, powerful, and packed with cool features under the hood, yet simple to get started. It supports multiple vector databases, numerous LLM and Embedding providers, and includes a fully-offline mode that runs the models locally. Abstractions make it easy to scale your chains in complexity and add steps to any workflow to suit your needs.

## Repository Structure

```sh
└── Augmenta/
    ├── TODO.md
    ├── chat.py
    ├── classes.py
    ├── config.py
    ├── constants.py
    ├── documents
    │   └── 12AngryMen.pdf
    ├── embed.py
    ├── evaluation_workflow.jpeg
    ├── helpers.py
    ├── manifest.json
    ├── models.py
    ├── notebooks
    │   ├── agents.ipynb
    │   ├── gradio.ipynb
    │   ├── notebook.ipynb
    │   ├── notebook_rag.ipynb
    │   ├── pipelines.ipynb
    │   ├── retriever.ipynb
    │   └── vectors.ipynb
    ├── rag.py
    ├── reformatter.py
    ├── requirements.txt
    ├── sample.txt
    └── settings.json
```

## Modules

### File Summary

| File | Summary |
| --- | --- |
| [config.py](config.py) | Defines special settings for the Chatbot, as well as custom configurations for RAG metadata mapping and filtering a vector database by topic. |
| [settings.json](settings.json) | The main configuration file for the Chatbot. Set up the Chat and RAG specifications here, as these are the settings used to instantiate chats, as well as get inputs and parameters for the vectorization and retrieval. |
| [embed.py](embed.py) | Module to load and index documents into vector stores for further processing. Supports loading from URLs, local files, and text files using various loaders like WebBaseLoader, ArxivLoader, UnstructuredFileLoader, and TextLoader. Chroma and FAISS vector stores are used to store the vectors obtained through the Embedder model. |
| [models.py](models.py) | Define and create instances of models using different APIs (Together, OpenAI, Anthropic, Ollama, LMStudio) from this file. The MODEL_DICT stores the up to date list of supported models along with useful metadata. |
| [requirements.txt](requirements.txt) | The essential packages use the Langchain framework and their integrations. `faiss-cpu` / `chromadb` are the main dependencies for the vector database. |
| [constants.py](constants.py) | Defines the templates for system messages and the different chains. |
| [chat.py](chat.py) | Implements an interactive chatbot that can operate in two modes: a standard chat mode and a RAG (Retrieval-Augmented Generation) mode. In standard chat mode, the chatbot responds to user input using a configurable LLM. In RAG mode, the chatbot indexes documents and uses a vector database to retrieve the most contextually appropriate sources to generate responses with high ground truth. |
| [rag.py](rag.py) | Module to run the Retrieval-Augmented Generation (RAG) pipeline, taking a language model (llm), a retriever, and a template to create a chain. This pipeline is used to retrieve relevant documents, format them, perform in-between steps like generate summaries or evaluate the excerpts, then generate a high quality response using the appropriate context. |
| [manifest.json](manifest.json) | In this repository, the `manifest.json` file serves as a reference manifest for database collections. It specifies a database with an ID, collection name, and metadata including embedding model, search method, chunk size, and input documents. |
| [classes.py](classes.py) | Defines schemas for Chat, RAG, and Hyperparameter settings. Custom classes within the Config object for validating and adjusting settings dynamically. |
| [helpers.py](helpers.py) | Utility functions, including dealing with reading and updating the manifest.json file. Other functions include saving response strings to markdown files, cleaning text, formatting documents for context input, and checking existence for vector databases. |

## Getting Started

### System Requirements:

* **Tested on Python**: `version 3.11.7`

### Installation

#### From Source

1. Clone the Augmenta repository:
```console
$ git clone https://github.com/TanGentleman/Augmenta.git
```
2. Enter the project directory (venv activation for Linux/MacOS):
```console
$ cd Augmenta
$ python -m venv .venv
$ source .venv/bin/activate
```
3. If on Windows, activate the venv using:
```console
$ .venv\Scripts\activate.bat
```
4. Install the dependencies:
```console
$ pip install -r requirements.txt
```

### Usage

#### Setting up the .env file

To run Augmenta, you'll need to set up a `.env` file with the following variables:

* `VECTOR_DB_URL`: the URL of your vector database
* `LLM_API_KEY`: your API key for the language model
* `EMBEDDING_MODEL`: the name of the embedding model to use

Create a new file named `.env` in the root of the project directory. Use example-env.env or add the following contents:
```txt
OPENAI_API_KEY=""
TOGETHER_API_KEY=""
ANTHROPIC_API_KEY=""
LANGCHAIN_API_KEY=""
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="Augmenta"
```
Replace the placeholders with your own values. Models set in settings.json will require an API key to authorize with the respective provider, but none are required, as the project can run fully locally using LMStudio or Ollama as a backend inference server.

Run Augmenta using the command below:
```console
$ python chat.py
```
Optional flags: `-np` (non-persistent), `-rag` (rag mode). Append a prompt to the command to start the chat with a specific prompt. For example, if loading inputs that contain hundreds of receipes, you can quickly get a response saved to response.md without having a persistent chat:
```console
$ python chat.py -rag -np "What are the ingredients I would need for the tacos and enchiladas? Can I get all the ingredients from the local ALDI?"
```

## Project Roadmap

- [X] `Finally make this README!`
- [ ] `YAML implementation for settings.json`
- [ ] `Integration with document ingestion APIs (Vectara, AI21Labs, local Cohere server)`

## Contributing

Contributions are more than welcome! The best way to do so is creating an issue with the steps to reproduce the issue.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
