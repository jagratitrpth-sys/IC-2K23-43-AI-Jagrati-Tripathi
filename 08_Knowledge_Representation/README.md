# Lab 4: Knowledge Representation, Rule-Based Systems, Semantic Networks

## Problem Statement
Implement systems to represent logical knowledge and infer new information. The lab requires demonstrating how an AI agent can use a set of facts and rules to deduce conclusions, and how it can use a network graph to inherit properties.

## Concepts Implemented

### 1. Rule-Based System (Forward Chaining)
A rule-based expert system uses a knowledge base of "If-Then" rules. 
* **Forward Chaining:** Starts with a known set of facts and triggers rules whose `IF` conditions match the facts. The `THEN` conclusions are added to the known facts. This loops until no new rules can be triggered.
* *Example in code:* The system knows an animal `has_hair`, `has_claws`, and `has_sharp_teeth`. It first deduces `is_mammal`, and then uses that new fact to deduce `is_carnivore`.

### 2. Semantic Networks
A semantic network represents semantic relations between concepts in a network. It is often used as a form of knowledge representation using a directed or undirected graph.
* **Inheritance:** Nodes inherit properties from their parent nodes via `is_a` relationships.
* *Example in code:* A `Labrador` inherits the `sound: bark` property from `Dog`, the `gives: milk` property from `Mammal`, and the `needs: food` property from `Animal` through recursive lookup.

## Expected Output
```text
--- 1. Rule-Based System (Forward Chaining) ---
Initial Facts: {'has_sharp_teeth', 'has_hair', 'has_claws'}
Rule triggered! Inferred new fact: is_mammal
Rule triggered! Inferred new fact: is_carnivore
Final Knowledge Base: {'is_carnivore', 'has_sharp_teeth', 'has_hair', 'has_claws', 'is_mammal'}

--- 2. Semantic Network (Inheritance) ---
  Querying: What does Labrador 'color'?
Result for Labrador color: golden

  Querying: What does Labrador 'gives'?
  -> Labrador doesn't explicitly store this. Checking parent 'Dog'...
  Querying: What does Dog 'gives'?
  -> Dog doesn't explicitly store this. Checking parent 'Mammal'...
  Querying: What does Mammal 'gives'?
Result for Labrador gives: milk