def rule_based_system():
    print("--- 1. Rule-Based System (Forward Chaining) ---")
    
    # Initial facts known to the system
    facts = {"has_hair", "has_claws", "has_sharp_teeth"}
    print(f"Initial Facts: {facts}")

    # Rules defined as: (Condition set, Result fact)
    rules = [
        ({"has_hair"}, "is_mammal"),
        ({"gives_milk"}, "is_mammal"),
        ({"has_feathers"}, "is_bird"),
        ({"is_mammal", "has_claws", "has_sharp_teeth"}, "is_carnivore"),
        ({"is_carnivore", "has_black_stripes"}, "is_tiger"),
    ]

    new_facts_added = True
    while new_facts_added:
        new_facts_added = False
        for conditions, result in rules:
            # If all conditions are met and the result isn't already a fact
            if conditions.issubset(facts) and result not in facts:
                facts.add(result)
                print(f"Rule triggered! Inferred new fact: {result}")
                new_facts_added = True

    print(f"Final Knowledge Base: {facts}\n")


def semantic_network():
    print("--- 2. Semantic Network (Inheritance) ---")
    
    # Representing knowledge as a graph using nested dictionaries
    network = {
        'Animal': {'can': 'move', 'needs': 'food'},
        'Mammal': {'is_a': 'Animal', 'has': 'hair', 'gives': 'milk'},
        'Dog': {'is_a': 'Mammal', 'has': 'tail', 'sound': 'bark'},
        'Labrador': {'is_a': 'Dog', 'color': 'golden'}
    }

    def query_property(entity, relation):
        """Recursively traverses the semantic network to find inherited properties."""
        print(f"  Querying: What does {entity} '{relation}'?")
        
        # Base case: The property exists directly on the entity
        if entity in network and relation in network[entity]:
            return network[entity][relation]
            
        # Recursive case: The entity inherits from a parent category ('is_a')
        if entity in network and 'is_a' in network[entity]:
            parent = network[entity]['is_a']
            print(f"  -> {entity} doesn't explicitly store this. Checking parent '{parent}'...")
            return query_property(parent, relation)
            
        return "Unknown"

    # Direct query
    print(f"Result for Labrador color: {query_property('Labrador', 'color')}\n")
    
    # Inherited query (Labrador -> Dog -> Mammal)
    print(f"Result for Labrador gives: {query_property('Labrador', 'gives')}\n")
    
    # Deeply inherited query (Labrador -> Dog -> Mammal -> Animal)
    print(f"Result for Labrador needs: {query_property('Labrador', 'needs')}\n")


if __name__ == "__main__":
    rule_based_system()
    semantic_network()