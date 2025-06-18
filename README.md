# LLM Agent Scaffolder

`llm-agent-scaffolder` is a command-line tool designed to rapidly generate a complete boilerplate for building a custom, multi-agent framework powered by Google's Gemini LLM.

It sets up the entire project structure, including directories for agents, tools, orchestration, and LLM clients, allowing you to go from an idea to a working agent framework in minutes.

## Features

- **Full Project Initialization**: Creates a complete, ready-to-use project structure with a single command.
- **Component Scaffolding**: Easily add new, empty `Agent` or `Tool` files to your existing project.
- **Modular by Design**: The generated boilerplate is organized into logical components (agents, tools, orchestrator) for maximum extensibility.
- **Gemini-Ready**: Includes a pre-configured client for interacting with the Google Gemini API.

## Installation

```bash
pip install llm-agent-scaffolder
UsageInitialize a New ProjectTo create a brand new agent framework project, run:create-simple-agent init --name MyAgentProject
This will create a new directory MyAgentProject/ with the complete boilerplate.Add New ComponentsOnce inside a project directory, you can easily add new components:Create a new Tool:create-simple-agent new tool --name MyFileParser
Create a new Agent:create-simple-agent new agent --name DataAnalysisAgent
Generated Project StructureThe init command generates the following structure:MyAgentProject/
├── agents/
│   ├── base_text_agent.py
│   ├── text_conversational_agent.py
│   └── text_tool_execution_agent.py
├── llm_clients/
│   ├── base_prediction_client.py
│   └── gemini_prediction_client.py
├── orchestrator/
│   ├── text_agent_graph_loader.py
│   └── text_agent_orchestrator.py
├── tools/
│   ├── base_tool.py
│   └── sample_tool.py
├── utils/
│   ├── data_models.py
│   └── logger.py
├── config.py
├── main.py
├── text_agent_configs.json
└── text_agent_workflows.json
