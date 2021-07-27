from .utilities import create_trivial_argument, get_endpoints, prob

import networkx as nx
import pandas as pd
import numpy as np

from google.colab import widgets

# INTERPRETING SCORING TABLE
is_sub_argument = lambda arg1, arg2: \
 set(arg1.observations.items()) <= set(arg2.observations.items()) \
 and set(arg1.nodes) <= set(arg2.nodes) \
 and set(arg1.edges) <= set(arg2.edges)

def read_scoring_table(model, target, evidence, scoring_table, 
                       verbose=False, interactive = False):
  # Find baseline entry and drop it from table
  baseline_entry = scoring_table.iloc[0]
  score = baseline_entry["score"]

  scoring_table = scoring_table.drop(0, axis=0)

  # Filter rows that apply to this input
  filter_fn = lambda row : \
    (set(row['observations'].items()) <= set(evidence.items())) and \
    (set(evidence.keys()).isdisjoint(set(row["argument"].nodes) - set(row["observations"].keys())))
    
  mask = scoring_table.apply(filter_fn, axis=1)
  scoring_table = scoring_table[mask]

  # Filter less specific rows in favor of more specific joint rows
  minimal_rows = []
  for index, row in scoring_table.iterrows():
    for argument in scoring_table["argument"]:
      if argument == row["argument"]: continue
      if is_sub_argument(row["argument"], argument): break
    else:
      minimal_rows.append(row)
  
  scoring_table = pd.DataFrame(minimal_rows)

  # Announce baseline in verbose mode
  if verbose:
    arguments = [create_trivial_argument(target[0])]
    tb = widgets.TabBar([str(i) for i in range(len(scoring_table)+2)]) if interactive else None
    i = 0
    with tb.output_to(i, select=True) if interactive else nullcontext():
      i+=1
      p = 1. / (1./np.exp(score) + 1.)
      print("")
      print(baseline_entry["explanation"])
      if interactive: draw_model(model, arguments)

  # Process each relevant argument
  for index, row in scoring_table.iterrows():
    # Add precomputed score contribution to current score
    old_score = score
    score += row['score']

    if verbose:
      with tb.output_to(i, select=False) if interactive else nullcontext():
        i+=1
        for line in row["explanation"]: print(line)
                
        old_p = 1. / (1./np.exp(old_score) + 1.)
        p = 1. / (1./np.exp(score) + 1.)
        explanation = model.explanation_dictionary[target]["explanation"]
        print(f"This consideration changes the probability that {explanation} from {old_p*100:0.2f}% to {p*100:0.2f}%")
        if interactive: draw_model(model, [row['argument']])
        else: print("")
        arguments.append(row['argument'])


  if verbose:
    with tb.output_to(i, select=False) if interactive else nullcontext():
      p = 1 / (1 + 1 / np.exp(score))
      explanation = model.explanation_dictionary[target]["explanation"]
      
      print(f"After taking into account these {i} arguments, we conclude that ")
      print(f"the probability that {explanation} is {p*100:0.2f}%")

      p = prob(model, target, dict(evidence))
      print(f"For comparison, the estimation of the probability using message passing is {p*100:0.2f}%")
      if interactive: draw_model(model, arguments)

  return score

# DRAWING GRAPHS WITH HIGHLIGHTED PATHS
swap = lambda edge : (edge[1], edge[0])

def get_target(argument):
  for node in argument.nodes:
    if argument.in_degree(node) == 0:
      return node
  
  assert False, "The argument loops on itself and has no conclusion"

def draw_model(model, arguments=[], argument=None):
  G = model
  df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
  for row, data in nx.shortest_path_length(G):
    for col, dist in data.items():
        df.loc[row,col] = dist
  df = df.fillna(df.max().max())
  pos = nx.kamada_kawai_layout(G,dist=df.to_dict())

  if argument is not None:
    arguments = [argument]
  
  if len(arguments) > 0:
    nodes_to_color = []
    edges_to_color = []
    sources = []
    
    for argument in arguments:
      nodes_to_color += [node for node in argument if node in model]
      edges_and_swapped_edges = list(argument.to_undirected().edges) + \
                                [(n2,n1) for (n1,n2) in argument.to_undirected().edges]
      edges_to_color += [(node1, node2) 
                         for (node1, factor1) in edges_and_swapped_edges 
                         for (factor2, node2) in edges_and_swapped_edges
                         if isinstance(node1, str) and isinstance(node2, str) 
                         and node1 != node2 and factor1 == factor2]
      sources += get_endpoints(argument)
      
    target = get_target(argument)

    node_color_map = ['orange' if node == target else
                      'blue' if node in sources else 
                      'green' if node in nodes_to_color else 
                      'gray' 
                      for node in model]
        
    edge_color_map = ['green' if edge in edges_to_color 
                              or swap(edge) in edges_to_color 
                      else 'gray' 
                      for edge in model.edges]
  else:
    node_color_map = 'yellow'
    edge_color_map = 'green'

  nx.draw(G, pos, 
          node_size = 800,
          width = 3,
          with_labels=True, 
          node_color = node_color_map,
          edge_color = edge_color_map,
          font_weight = 'bold')
