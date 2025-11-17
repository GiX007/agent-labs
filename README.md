# Agent Labs

A carefully curated collection of labs, notes, and graded assignments from multiple DeepLearning.ai courses focused on prompt engineering, LLM systems, agentic workflows, LangChain/LangGraph, multi-agent frameworks, RAG, and advanced system designs.  
This repository serves as a practical foundation for for understanding all the basics before starting building real agentic applications, from simple prompt pipelines to multi-agent orchestration and production-ready AI systems.

---

## Contents

- **00_agentic_ai_course:** Foundations of agentic AI, covering core design patterns such as reflection, tool use, planning for multi-agent workflows, and evaluation methods.  
- **01_prompt_engineering:** Prompting techniques, best-practice patterns, and structured output strategies.  
- **02_building_systems_with_ChatGPT_API:** System design with OpenAI APIs, tool use, function calling, and evaluation.  
- **03_langchain:** Chains, tool integrations, workflow modularity, and system patterns.  
- **04_langGraph:** Graph-based agentic workflow design with nodes, edges, and stateful reasoning.  
- **05_crewAI:** Multi-agent collaboration using CrewAI (roles, tasks, tools, and handoffs).  
- **06_autoGen:** Multi-agent collaboration using AutoGen. 
- **07_rag_course:** Retrieval-Augmented Generation concepts— such as embeddings, retrieval functions, chunking, vector databases, and evaluation.

---

## Purpose

This repo is my playground and knowledge base for building modern agentic AI applications. Every module contributes toward developing scalable agents for real-world use cases.

---

## Knowledge Map

### Agentic AI Course
#### Core Concepts
- Agent = LLM + Tools + Memory + (optional) Planner
- Agent types: single-agent, multi-agent
- Execution modes: synchronous (step-by-step), asynchronous (parallel tasks)
- Tool types: API calls, LLM calls, code execution, search, retrieval
- Memory types: short-term, long-term
- Guardrails: defense mechanisms to limit bad actions, enforce rules, and keep the agent safe and reliable
- Feedback loops: self-critique, reflection, and refinement steps
- Agentic workflows: thought-action-observation loops
- Orchestration layer: the controller coordinating multiple agents, tools, and memory

#### Design Patterns
- Reflection (e.g., coder agent-critic agent) 
- Tool use via JSON schema  
- Planning (code-based is better than LLM-based)
- Multi-agent collaboration  

#### Evaluation
- Objective code tests  
- Subjective LLM-as-judge (rubric, ideal answers)
- Human feedback (surveys, A/B testing)
- End-to-end evaluation  
- Component-level evaluation  
- Error analysis to improve latency, cost, accuracy (=quality and correctness) 

#### Development Process Summary 
- build end-to-end agentic workflow 
- examine outputs and traces 
- build evals and compute metrics 
- perform error analysis 
- build component-level evals 
- improve individual components

#### Examples
- Invoice processing  
- Essay: search → draft → revise  
- Sales visualization with code critic  
- SQL/chart/essay generation (with reflection) 
- Research agent  
- Marketing team (copywriter, SEO specialist, social media manager)
- Customer service agent 
- Email agent

---

### Prompt Engineering for LLMs
#### Core Principles
- **Clear & specific instructions**: delimiters, define clear tasks, capture edge cases, structured output, condition checks, few-shot prompting
- **Give the model time to think**: specify steps, ask for step-by-step reasoning, private reasoning before answer (ask to work on its own solution before rushing to a conclusion), allow hidden internal steps that guide thinking without showing them in the final answer (“Think through the solution step by step internally, but only give me the final answer.”
)

#### Common Task Patterns
- Marketing text generation (length, format, focus), e.g., product descriptions
- Review summarization  
- Sentiment/emotion analysis  
- Information extraction, e.g., topics from text
- Alert creation, e.g., flag negative reviews or if a topic is mentioned
- Transformations: translation, tone, format conversion, grammar checks  
- Customer-service workflows, e.g., send apology email if negative sentiment detected
- Multi-turn chatbots with memory (e.g., pizza order bot)

---

### Building With ChatGPT API
#### LLM Types
- **Base LLM**: Predicts next word 
- **Instruction-tuned LLM**: Follows instructions
- Roles: system (sets the tone/behavior), user (prompts), assistant (responses) 
- Token limits = input context + output completion

#### Development Speed
- Prompt-based: hours
- Supervised Learning: 6+ months (data + training + deployment)

#### Prompt Chaining
- Break down tasks  
- Manage context and reduce cost  
- Human-in-loop  
- State tracking  
- Tool integration  

#### LLM Cost Optimization
- Shorter prompts
- Lower-cost models for simple tasks
- Reduce output length
- Use embeddings + retrieval for knowledge-intensive tasks

#### How to Choose an LLM
- Capabilities (reasoning, coding, knowledge, creativity)
- Context length
- Cost
- Latency
- Training data cutoff date

#### Techniques to Improve LLM Outputs
- Temperature and top-p tuning  
- Output constraints (formatting, length)
- Few-shot prompting
- Chain-of-thought prompting (add "Think step-by-step" to prompt)
- Encourage self-reflection (e.g., "Before answering, check your work.")
- Encourage reasoning (e.g., "Explain your reasoning by breaking down the problem: step1:..., step2....")
- Citations for hallucination reduction

#### Examples
- Write a poem with specific style keywords and length
- Classify customer queries to handle different cases
- Use moderation to detect policy violations in user input and LLM's output
- Detect prompt injections  
- build simple knowledge base using JSON data, dict of dicts

#### Full Workflow Example with Moderation and Validation
1. Moderate user input  
2. Extract products  
3. Lookup data for product's info 
4. Generate answer  
5. Moderate output  
6. Validate correctness (check by LLM) 
7. Respond

---

### LangChain — LLM Application Development
#### Core Components
- Template prompts
- LLMs  
- Tools  
- Retrievers  
- Output parsers (convert LLM output into structured format; use ResponseSchema + StructuredOutputParser) 
- Memory  

All components are combined into **Chains** defining thought-action-observation loops.

#### Chains
- SimpleSequentialChain (single input/output)
- SequentialChain (multiple inputs/outputs)  
- RouterChain  
- LCEL (`prompt | llm | parser`)

#### Typical Use Cases
- Translation, tone change  
- Summaries, descriptions (of products, reviews, companies)
- Multilingual review → English → Summary → Follow-up response  
- Router-based problem solving (math, coding, general questions)
- Product Q&A with KB  
- artificial Q&A generation + evaluation  
- Simple agents with search/math/custom tools  

#### Full Workflow Example for Q&A Over Documents
1. Load documents (PDF, URL, YouTube, etc.)  
2. Split documents into chunks
3. Create embeddings for each chunk  
4. Store embeddings in a vector DB (e.g., Chroma) 
5. Retrieve relevant chunks using similarity search 
6. Optional: Compress / summarize chunks to reduce context size  
7. Pass retrieved chunks to LLM via Q&A chain (methods: stuff, map_reduce, refine, map_rerank)
8. Evaluate answer (optional: code check, LLM-as-judge, human feedback) 
9. Return final response to user

---

### LangChain Tools
#### Key Concepts
- Runnable protocol (to create chains that can be called like functions)
- Tool binding  
- Pydantic validation  
- Tagging vs extraction  

#### Tool Examples
- Weather  
- Web search + summarization  
- Joke generator  
- Entity extraction  
- Wikipedia  
- Routing with tools (e.g., wikipedia search tool, weather tool)
- Chatbots with memory  

---

### LangChain RAG
#### Pipeline
1. Load documents (PDF, URL, YouTube, etc.)  
2. Split into chunks  
3. Embed  
4. Store in vector DB (Chroma)  
5. Retrieve  
6. Compress (optional) as a trick to reduce doc size / improve relevance
7. Q&A chain: prompt → prompt template → retriever(chunks, embeddings, probably compression) LLM → answer

---

### LangGraph with Agents From Scratch
#### Core Concepts
- Graph-based control  
- Shared agent state (state vs memory) 
- Multi-agent orchestration  
- Human-in-loop nodes  
- Tavily search tool  

#### Examples
- Essay writer with reflection + criticism  
- Multi-step workflows with human-in-loop 

---

### CrewAI — Multi-Agent Framework
#### Key Elements
- Define **Agents → Tasks → Crew**  
- Tools on agent or task level  
- Memory  
- Async parallel execution  

#### Use Cases
- Essay research  
- Customer service support 
- Outreach campaigns (recognize high-value leads and create engaging messages to them) 
- Event planning (venue coordinator, logistics manager, and marketing&communication manager agents)
- Financial analysis agents (data analyst, trading strategy, risk management, and trade advisor agents)
- Job filtering + interview prep  

---

### Autogen — Multi-Agent Conversations
#### Concepts / Examples
- Writer vs critic reflection loops  
- Multi-agent discussion (e.g., stand-up comedy brainstorming) 
- Code writer + executor  
- Cost reporting (significant when in production) 
- Blog writing w/ SEO + ethics reviewers 
- Stock evaluation  

---

### RAG Course
#### Why RAG?
- Private data  
- Real-time info  
- Lower cost (computational and context size)
- Better accuracy
- Fewer hallucinations  

#### RAG Components
- Retriever: fetch relevant docs from knowledge base
- (optional) Compressor: reduce doc size while retaining info
- (optional) Evaluator: assess doc relevance/quality
- Knowledge base: vector DB, SQL DB, NoSQL DB, file system, APIs
- RAG system = Retriever + (optional) Compressor + (optional) Evaluator + knowledge base → relevant docs
- Process: User query → RAG system → augmented prompt → LLM → answer
- Types of RAG systems: simple RAG, RAG with compression, RAG with evaluation, agentic RAG
#### Agentic RAG
- prompt → Router LLM → [KB → Retriever → relevant docs → LLM evaluator] → augmented prompt → LLM + Citations → response

#### Retrieval Methods 
- Keyword search: TF-IDF, BM25  
- Semantic search: embeddings  
- Metadata filtering (not a search by itself)  
- Hybrid search: (keyword search + filtering) + (semantic search + filtering) → combined results (RRF)

#### Retriever Evaluation
- Precision, recall  
- Top-k, mAP, MRR

#### Retriever Optimizations
- Query rewriting (prompt → llm query rewriter → optimized query → retriever → parser) 
- Cross-encoder reranking (produce optimal doc order)

#### Production
- Latency, throughput  
- Memory & compute usage 
- Monitoring (Phoenix) output quality, retriever performance, system health, user satisfaction
- Feedback loops for continuous improvement, cost management, security, privacy, versioning, testing and validation, deployment (blue/green, canary releases)

---

### Tools & Platforms
- Tavily search tool 
- frameworks: LangChain, LangGraph, CrewAI, Autogen
- Together.ai  
- Weaviate  
- Chroma  
- Phoenix  

---

### Key References
- DeepLearning.AI: https://www.deeplearning.ai  
- LangChain: https://python.langchain.com  
- LangGraph: https://langchain-ai.github.io/langgraph  
- CrewAI: https://docs.crewai.com  
- AutoGen: https://microsoft.github.io/autogen  
- Chroma: https://docs.trychroma.com  
- Weaviate: https://weaviate.io/developers/weaviate   
- Together.ai: https://together.ai
- Phoenix: https://phoenixmonitoring.ai
- OpenAI: https://platform.openai.com/docs/models
- Tavily: https://tavily.com

---

## Vision

This repository will continue to grow into a modular library of agentic patterns, a practical starting point for building production-ready LLM systems, autonomous workflows, and multi-agent solutions.
