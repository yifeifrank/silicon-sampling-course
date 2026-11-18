# Silicon Sampling Course · 硅样本课程

This demo uses language models to simulate a three-advisor discussion about a loss-framed choice, followed by an independent decision from each advisor.

本示例使用语言模型模拟三位顾问围绕损失框架下的选项展开讨论，随后分别征询每位顾问的决定。

- [English notebook](notebooks/prospect_theory_experiment_en.ipynb)
- [中文笔记本](notebooks/prospect_theory_experiment_zh.ipynb)

## Course reading · 课程阅读

- [Chapter 9. Social Simulation with Large Language Model Agents (English)](readings/social-simulation-with-llm-agents.md)

## Run the demo · 运行方法

Install the dependencies in a Python 3.10 or later environment and open JupyterLab.

在 Python 3.10 或更新版本的环境中安装依赖，然后打开 JupyterLab。

```bash
python -m pip install jupyterlab langchain-core langchain-openai langgraph pydantic typing-extensions
jupyter lab
```

Open either notebook and run the cells in order. At the setup cell, enter your API key in the hidden prompt and provide the name of a chat model that supports structured output and temperature settings. You can also set `OPENAI_API_KEY` and `OPENAI_MODEL` before starting JupyterLab. Set `OPENAI_BASE_URL` if your provider uses a custom OpenAI-compatible endpoint. A full run makes 33 model calls.

任选一个笔记本，按顺序运行单元格。在配置单元格中，通过隐藏输入框填写 API 密钥，并输入支持结构化输出和温度设置的聊天模型名称。也可以在启动 JupyterLab 前设置 `OPENAI_API_KEY` 和 `OPENAI_MODEL`。如果供应商使用自定义的 OpenAI 兼容端点，请设置 `OPENAI_BASE_URL`。完整运行会调用模型 33 次。
