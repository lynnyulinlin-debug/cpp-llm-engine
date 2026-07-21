---
layout: home

hero:
  name: cpp-llm-engine
  text: Build a C++ LLM Inference Engine
  tagline: From the first runnable baseline to a verifiable inference pipeline
  actions:
    - theme: brand
      text: Open the English tutorial entry
      link: /en/README
    - theme: alt
      text: Read the guide
      link: /en/guide

features:
  - title: English navigation first
    details: English readers can start from the entry pages, then move into the main line when ready.
  - title: Route-separated
    details: The site uses / and /en/ as separate routes so language choice stays visible.
  - title: Phase-based rollout
    details: We translate entry pages first, then expand to chapter intros and summaries.
---

## What this is

This is the English landing page for the tutorial. It is intentionally lightweight: the goal is to let English readers find the project, understand the reading path, and reach the first entry points quickly.

## Engine Layer

The tutorial focuses on the engine layer first: the part that turns a static model into a runnable runtime with clear boundaries for tensor representation, weight loading, operator execution, decoder structure, runtime state, KV cache, quantization, backend adaptation, testing, and benchmarking.

Think of Python inference frameworks as the "control tower" and the C++ inference engine as the "engine":

- the control tower organizes requests and workflows
- the engine executes the hot path, manages memory and state, and determines latency and throughput

If you want the fuller explanation, open [training vs inference frameworks](/en/reference/framework_comparison).
If you want the shortest route map, start from [guide](/en/guide).

## What you will see

- The main tutorial entry
- The reading guide
- Foundation and chapter landing pages in English
- The Chinese main line when you need the full walkthrough

## Who this is for

- Readers who want an English entry point first
- Contributors who need a clear route map before translating more pages
- Anyone comparing the Chinese main line with the English navigation layer
