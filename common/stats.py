import networkx as nx
from common.problem import Formula
from envs import ResUNSAT
from common.utils import get_opt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def symbolize_clause(clause):
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

def build_formula_graph(env, formula: Formula):
    G = nx.Graph()
    # There's an edge between two clauses iff they share a variable
    N = len(formula.clauses)
    env.clause_set = set()
    for i in range(N):
        for j in range(i+1, N):
            c1 = set([l for l in formula.clauses[i]])
            c2 = set([l for l in formula.clauses[j]])
            res = env.resolve(c1, c2)
            if res is not None:
            # if len(c1.intersection(c2)) > 0:
                G.add_edge(i, j)
                G.add_edge(j, i)

    return G

def compute_diameters(env):
    modes = ["train", "val", "test"]
    data_dict = env.dataset
    for mode in modes:
        formulas = data_dict[mode]["problems"]
        all_diameters = [formula_diameter(f) for f in tqdm(formulas)]
        print(f"Mode: {mode}, Avg. diameter: {np.mean(all_diameters)}, Min diameter: {np.min(all_diameters)}, Max diameter: {np.max(all_diameters)}")

def draw_smallest_graph(env):
    data_dict = env.dataset
    problems = data_dict["train"]["problems"]
    # Get formula with least number of clauses
    lens = [len(p.clauses) for p in problems]
    idx = np.argmin(lens)
    f = problems[idx]
    G = build_formula_graph(env, f)
    node_attrs = dict(node_shape="s",  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.3'))
    nx.draw_circular(G, with_labels=True, font_size=10, font_color="black", edge_color="gray", **node_attrs)
    plt.show()

def formula_diameter(formula: Formula):
    G = build_formula_graph(formula)
    return nx.diameter(G)

opt = get_opt()
env = ResUNSAT(opt, shuffle=False)

draw_smallest_graph(env)