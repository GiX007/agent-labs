# Prompt Engineering Playbook

## Principles, Use Cases, and Reusable Templates

---

## Overview

This playbook is structured into:

1. **Principles** – high-level rules (why prompting works)
2. **Tactics / Patterns** – how principles are applied
3. **Use Cases** – what you actually do with prompts in practice

**Built to design the right prompt for any use case.**

---

## 🔹Principle 1: Write Clear & Specific Instructions

Reduce ambiguity. Increase control. Improve reliability.

---

### Tactic 1: Role Assignment (Role Prompting)

**Template** 

You are an expert in '{field}'. Your goal is to '{objective}'.

**Description**

Sets domain expertise and expected reasoning depth.

**Example**

You are an expert ML engineer. Your goal is to design a RAG-based QA system.

---

### Tactic 2: Use Delimiters

**Template** 

'{input}'.

**Description**

Separates instructions from data and clearly defines context boundaries.

**Example**

Summarize the following text: '{article}'.

---

### Tactic 3: Ask for Structured Output

**Template**

Return the output in the following format:
  - Section 1:
  - Section 2:

**Description**  

Forces for consistent and predictable structured outputs.

**Example**

Return the output as:
 - Problem
 - Proposed Solution
 - Risks

---

### Tactic 4: Condition Checking

**Template**

Before answering, verify that:
  1. '{condition}'
  2. '{condition}'
If not satisfied, explain why.

**Description**

Reduces hallucinations and enforces constraints.

**Example**

Verify that the answer uses only the provided '{context}'.

---

### Tactic 5: Few-Shot Prompting

**Template**

Example: ...
Input: ...
Output: ...

Now do:
Input: {new input}

**Description**

Teaches the model the desired pattern through examples.

**Example**

Input: "Great service"
Output: Positive

Now classify:

Input: "Terrible experience"

---

## 🔹 Principle 2: Give the model time to think

Improve reasoning quality for complex tasks or multi-step tasks.

---

### Tactic 1: Explicit Step Decomposition

**Template**

Follow these steps:
1. ...
2. ...
3. ...

**Description**

Encourages structured reasoning by breaking tasks into steps.

**Example**

1. Identify user intent
2. Retrieve relevant documents
3. Generate a grounded answer

---

### Tactic 2: Delayed Final Answer

**Template**

First work out the solution. Then provide the final answer. (Alternative: “Think step by step.”)

**Description**

Prevents shallow or rushed conclusions.

**Example**

Analyze tradeoffs first, then recommend a model architecture.

---

## 🔹 Core Use Cases

What you actually do with prompts.

---

## Use Case 1: Summarization

### Focused Summary

**Template**

Summarize with focus on '{aspect}'.

**Description**  

Directs attention to a specific dimension of the content.

**Example**

Summarize focusing on business impact.

---

### Extraction 

**Template**

Extract only:
  - '{item 1}'
  - '{item 2}'

**Description**  
Returns specific information without paraphrasing.

**Example**

Extract all deadlines and action items.

---

### Length Control (General Use)

**Template**

Limit the output to '{n}' bullet points / '{n}' words.

**Description**  

Controls verbosity and response size.

**Example**

Limit your answer to one sentence.

---

## Use Case 2: Inferring Information

### Sentiment Analysis

**Template**

Classify the sentiment as Positive, Neutral, or Negative.

**Description**  

Infers overall sentiment from text.

**Example**

From the below '{review}', determine whether the sentiment is positive or negative.

---

### Emotion Detection

**Template** 

Identify the dominant emotions expressed.

**Description**  

Detects emotional signals beyond simple sentiment.

**Example**  

Identify the dominant emotions in the '{message}'.

---

### Topic Classification

**Template**: Identify the main topics discussed.

**Description**  

Extracts high-level themes or categories.

**Example**  

Identify the main topics discussed in the '{text'}.

---

## Use Case 3: Transformation

### Format Conversion

**Template**

Convert the following from '{format A}' to '{format B}'. 

**Description**  

Transforms content across structured formats.

**Example**

Convert this JSON into an HTML table.

---

### Language Translation

**Template**: Translate the following '{text}' to '{language}'.

**Description**  

Performs multilingual translation.

**Example**  

Translate the following '{text}' to Spanish.

---

### Tone Transformation

**Template**

Rewrite with a {tone} tone.

**Description**  

Adjusts style while preserving meaning.

**Example**

Rewrite with a professional and concise tone.

---

## Use Case 4: Data Generation

### Synthetic Data Generation

**Template**

Generate '{n}' high-quality examples of '{task}'.

**Description**  

Creates synthetic data for training, testing, or prototyping.

**Example**

Generate 50 user questions for a travel booking chatbot.

---

## Use Case 5: Retrieval-Augmented Generation (RAG)

**Template / Example**

Using only the '{context}' below, answer the '{question}'.
Context: '{docs}'
Question: '{question}'

**Description**  

Grounds the response strictly in retrieved external knowledge.

## Use Case 6: Evaluation (LLM as a judge)

**Template**

Evaluate the following '{output}' using these criteria:
 - Accuracy
 - Clarity
 - Completeness
Score from 1–5 and explain.

**Description**  

Uses the model to evaluate another model’s output.

**Example**  

Evaluate the generated summary for factual correctness.

---

## Use Case 7: Iterative Refinement

### Output Too Long

**Template**

Make the response more concise. Limit to '{n}' points.

**Description**  

Reduces verbosity.

### Wrong Focus

**Template**

Rewrite focusing only on '{specific aspect}'.

**Description**  

Redirects attention to the correct dimension.

### Needs Expansion

**Template**: 

Expand on '{section}' with concrete examples.

**Description**  

Adds depth where needed.

---

## How to Use this Playbook

1. Choose a **principle**
2. Select a **use case**
3. Replace `{task}` / `{topic}` / `{context}`
4. Paste into ChatGPT
5. Iterate with refinement prompts

---

## Sources

### Core Prompt Engineering Guides

- DeepLearning.AI – Prompt Engineering for Developers  
  https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/
- OpenAI Prompt Engineering Guide  
  https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic Prompt Engineering Guide  
  https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview

---

### Prompt Template Galleries & Examples

- LangChain Prompt Hub   
  https://smith.langchain.com/hub

- Google Vertex AI Prompt Gallery   
  https://docs.cloud.google.com/vertex-ai/generative-ai/docs/prompt-gallery

- PromptBase  
  *(Community marketplace with many reusable prompt templates across use cases)*  
  https://promptbase.com/

---
