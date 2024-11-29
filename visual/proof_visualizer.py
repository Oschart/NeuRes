import networkx as nx

import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, width=800, height=600):
        pass

    def reset(self):
        pass

    def symbolize_clause(self, clause):
        expression_parts = []

        for lit in clause:
            if lit <= 26:
                letter = chr(abs(lit) + 96)  # 'a' corresponds to 1, 'b' corresponds to 2, and so on
            else:
                greek_letter = chr(abs(lit) + 944)  # Using Greek letters starting from 945 ('α' for 27, 'β' for 28, and so on)
                letter = f'{greek_letter}'
            
            if lit > 0:
                expression_parts.append(letter)
            elif lit < 0:
                expression_parts.append(f'{letter}̅')

        expression = ' ∨ '.join(expression_parts)
        if len(expression) == 0:
            expression = '⊥'
        return expression

    def symbolize_proof(self, proof):
        symb_step = lambda s: [self.symbolize_clause(c) for c in s]
        # symbolized_proof = [tuple(symb_step(step)) for step in proof]
        symbolized_proof = proof
        return symbolized_proof

    def draw_proof(self, proof):
        self.reset()
        proof = self.symbolize_proof(proof)
        G = nx.DiGraph()
        first_app = {}
        for i, step in enumerate(proof):
            p1, p2, c = step
            if p1 not in first_app:
                first_app[p1] = 0
            if p2 not in first_app:
                first_app[p2] = 0
            first_app[c] = max(first_app[p1], first_app[p2]) + 1
            G.add_node(p1)
            G.add_node(p2)
            G.add_edge(p1, c)
            G.add_edge(p2, c)

        # For visualization purposes, layout the nodes in topological order
        # for i, layer in enumerate(nx.topological_generations(G)):
        #     for n in layer:
        #         G.nodes[n]["layer"] = i

        for n in G.nodes:
            G.nodes[n]["layer"] = first_app[n]
        
        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        # pos = nx.spring_layout(G, seed=42)
        # Visualize the trie
        node_attrs = dict(node_shape="s",  node_color="none", bbox=dict(facecolor="darkred", edgecolor='black', boxstyle='round,pad=0.5'))
        nx.draw(G, pos=pos, with_labels=True, font_size=16, font_color="white", font_weight="bold", edge_color="gray", **node_attrs)
        plt.show()
        

# Example usage
# proof = [((-19, -13, -3, 1, 21), (13, 20, 21), (-19, -3, 1, 20, 21)), ((-3, -1), (-19, -3, 1, 20, 21), (-19, -3, 20, 21)), ((-19, 3, 21), (-19, -3, 20, 21), (-19, 20, 21)), ((-11, 21), (-21, -19, -11), (-19, -11)), ((-10, 22), (-22, -15, 11), (-15, -10, 11)), ((-16, 15), (-15, -10, 11), (-16, -10, 11)), ((-10, -8, 12, 16), (-16, -10, 11), (-10, -8, 11, 12)), ((-21, 10, 20), (-10, -8, 11, 12), (-21, -8, 11, 12, 20)), ((-19, 20, 21), (-21, -8, 11, 12, 20), (-19, -8, 11, 12, 20)), ((-19, -11), (-19, -8, 11, 12, 20), (-19, -8, 12, 20)), ((12, 19, 20), (-19, -8, 12, 20), (-8, 12, 20)), ((-12, -8), (-8, 12, 20), (-8, 20)), ((-14, -1), (-18, -4, 14), (-18, -4, -1)), ((-12, 1, 11, 19), (-18, -4, -1), (-18, -12, -4, 11, 19)), ((-11, -4), (-18, -12, -4, 11, 19), (-18, -12, -4, 19)), ((4, 19), (-18, -12, -4, 19), (-18, -12, 19)), ((12, 19, 20), (-18, -12, 19), (-18, 19, 20)), ((-19, -18), (-18, 19, 20), (-18, 20)), ((8, 18), (-18, 20), (8, 20)), ((-8, 20), (8, 20), (20,)), ((-20, 4), (20,), (4,)), ((-11, -4), (4,), (-11,)), ((-20, 1, 11), (-11,), (-20, 1)), ((20,), (-20, 1), (1,)), ((-14, -1), (1,), (-14,)), ((-18, -4, 14), (-14,), (-18, -4)), ((4,), (-18, -4), (-18,)), ((8, 18), (-18,), (8,)), ((-12, -8), (8,), (-12,)), ((-3, -1), (1,), (-3,)), ((7, 14), (-14,), (7,)), ((-7, -6, -4, 3, 18), (7,), (-6, -4, 3, 18)), ((4,), (-6, -4, 3, 18), (-6, 3, 18)), ((-18,), (-6, 3, 18), (-6, 3)), ((-3,), (-6, 3), (-6,)), ((-7, 6, 17), (7,), (6, 17)), ((-6,), (6, 17), (17,)), ((-17, 12, 16), (17,), (12, 16)), ((-12,), (12, 16), (16,)), ((-16, -4, 22), (16,), (-4, 22)), ((4,), (-4, 22), (22,)), ((-16, 15), (16,), (15,)), ((-22, -15, 11), (-11,), (-22, -15)), ((15,), (-22, -15), (-22,)), ((22,), (-22,), ())]
# proof = proof[-10:]
# visualizer = Visualizer()
# visualizer.draw_proof(proof)
