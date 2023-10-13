#!/usr/bin/python
import sys
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)
"""
In this python file there are all the classes and function needed to solve the NBD Challenge 1.
    
It is possible to run this file as a python script with the pip module 'memory-profiler' to
check the memory usage for the following methods for checking a graph connectivity:
  - Bread First Search
  - Laplacian
  - Irreducibility
  
  This script is meant to be used with the mprof executable.
  To run you have to install 'memory-profiler', to do so 'pip install memory-profiler'.
  
  Now you can use the mprof executable on this script, as:
  'mprof run script.py'

  To check the available parameters just run 'python script.py -h'.
"""
import random
import numpy as np
import pandas as pd
from time import time
import networkx as nx
from tqdm.auto import tqdm
from bisect import bisect_left
from joblib import Parallel, delayed

# Typing handling modules
from typing import List, Dict, Union, Tuple, Iterable
from collections.abc import Callable

# Needed for some calculations
from math import log, log10
from numpy.linalg import matrix_power
from numpy.linalg import eig
from scipy.stats import linregress, bootstrap

# Plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout # For this pygraphviz is needed 




#############################################################
#                                                           #
#                   First assignment stuff                  #
#                                                           #
#############################################################


def erdos_renyi_graph(n: int,
                      p: float,
                      seed: int = None) -> nx.Graph:
    '''Create and initialize a graph following the Erdos Renyi Graph Model
    
    Args:
      - n (int): number of nodes.
      - p (float): a value between 0 and 1, it rappresents the probability for which an edge exists.
      - seed (int): Indicator of random number generation state. Default None.

    Return:
      - g (nx.Graph): The erdos-renyi graph.
    '''
    # Set the seed
    random.seed(seed)

    # Create a graph
    g = nx.Graph()
    
    # Adding all the nodes to the graph g
    g.add_nodes_from(list(range(1, n+1)))
    
    # Adding random edges using as probability p
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if random.uniform(0, 1) <= p: g.add_edge(i, j)
    
    return g



def check_with_irreducibility(g: nx.Graph) -> bool:
  """The function checks the connectivity of the passed graph using the
  Irreducibility method.

  Args:
    - g (nx.Graph): The graph we want to check.

  Returns:
    - bool : True if the graph is connected, otherwise False.
  """
  # Set A, n and the matrix variables
  A = nx.adjacency_matrix(g).todense().astype(np.uint16)
  n = g.number_of_nodes()
  matrix = np.identity(n)

  # Add the powers of A to the matrix variable
  for i in range(1, n):
    matrix += matrix_power(A, i)

  return np.all(matrix > 0)



def check_with_laplacian(g: nx.Graph,
                         d: int = 2) -> bool:
  """The function checks the connectivity of the passed graph using the
  Laplacian method.

  Args:
    - g (nx.Graph): The graph we want to check.
    - d (int): Number of decimal places to round to. If decimals is negative,
              it specifies the number of positions to the left of the decimal
              point. Default 2.

  Returns:
    - bool : True if the graph is connected, otherwise False.
  """
  # Create the diagonal matrix in which each element is the degree of the 
  # corresponding node
  D = list(dict(g.degree()).values()) * np.identity(g.number_of_nodes())

  # Get the Laplacian
  L = D - nx.adjacency_matrix(g).todense().astype(np.uint16)

  return np.sum(np.around(np.array(eig(L)[0]), d) == 0) == 1



def check_with_bfs(g: nx.Graph()) -> bool:
  """The function checks the connectivity of the passed graph using the
  Breadth-First-Search algorithm.

  Args:
    - g (nx.Graph): The graph we want to check.

  Returns:
    - bool : True if the graph is connected, otherwise False.
  """
  # Set start_node, visited and queue variables
  start_node = next(iter(g.nodes()))
  visited = set()
  queue = set([start_node])

  # Visiting the queue
  while queue:

    # Get the first node waiting in queue
    current = queue.pop()

    # Check if it was not visited before
    if current not in visited:

      # Added to the visited
      visited.add(current)

      # Get the neighbors and save them in the queue
      neighbors = g.neighbors(current)
      queue.update(neighbors)

  return len(visited) == len(g.nodes())



def time_complexity(functions: List[Callable[[nx.Graph], bool]],
                    nodes: List[int],
                    gtype: str,
                    p: float = 0.1,
                    d: int = 3,
                    n_jobs: int = -1,
                    M: int = 1000,
                    seed: int = None) -> Dict[str, Union[List[float], List[int]]]:
  """A function used to measure the time complexity for the passed functions.

  Args:
    - functions (List[Callable[[nx.Graph], bool]]): A list of function to check.
    - nodes (List[int]): The list of nodes.
    - gtype (str): The string describing the type of the graph.
    - p (float): The probability for an edge to exist, this is used for
                  erdos-renyi graph. Default 0.1.
    - d (int): Used for d-regular graph only, this is the number of edges
              for each node. Default 3.
    - n_jobs (int): The number of jobs for joblib.Parallel, it defines the
                    number of cores to use, if -1 it all available cores are used.
                    Default -1.
    - M (int): The number of simulation. Default 1000.
    - seed(int): Indicator of random number generation state. Default None.
  
  Returns:
    - temp (Dict[str, Union[List[float], List[int]]]): A dictionary with keys
                                                       the names of the passed 
                                                       functions plus the
                                                       current node value and
                                                       as values the
                                                       corresponding list of
                                                       times.
  """
  assert gtype == "erdos-renyi" or gtype == "r-regular", f"The 'type' passed is not handled, got {gtype}"
  
  def simulate(n: int,
               fun: Callable[[nx.Graph], bool]) -> float:
    if gtype == "erdos-renyi":
      g = erdos_renyi_graph(n=n, p=p, seed=seed)
    elif gtype == "r-regular":
      g = nx.random_regular_graph(n=n, d=d, seed=seed) 

    start = time()
    fun(g)
    end = time()

    return end-start

  temp = {}
  for fun in functions: temp[fun.__name__] = []
  for _ in tqdm(range(M)):
    for fun in functions:
      temp[fun.__name__] += Parallel(n_jobs=n_jobs)(delayed(simulate)(n=n, fun=fun) for n in nodes)

  temp['nodes'] = nodes*M
  temp['type'] = gtype

  return temp



def plot_time_complexity(n: int,
                         functions: List[Callable[[nx.Graph], bool]],
                         m: int = 1000,
                         p: float = 0.5,
                         d: int = 3,
                         n_jobs: int = -1,
                         save: bool = False,
                         seed: int = None) -> None:
  """A function that plots the result of the time complexity in:
    - normal scale.
    - log scale.
    - loglog scale.

  Args:
    - n (int): Number of nodes.
    - functions (Callable[[nx.Graph], bool]]): A list of function to check.
    - m (int): The number of simulation. Default 100.
    - p (float): The probability for the Erdos Renyi graph. Default 0.1.
    - d (int): The degree parameter for nx.random_regular_graph graph. Default 3.
    - n_jobs (int): The number of jobs for time_complexity, it defines the
                    number of cores to use, if -1 it all available cores are used.
                    Default -1.
    - save (bool): Save the measures dataframe in parquet format. Default False.
                   The dataframe will be saved as 'time_complexity.parquet' in
                   the current directory.
    - seed (int): Indicator of random number generation state. Default None.

  Returns:
    - None
  """
  sns.set()

  start = step = 10**(int(log10(n))-1)
  nodes = list(range(start, n+step, step))

  print("Calculating times for R-Regular graph...")
  df = pd.DataFrame(time_complexity(functions=functions,
                                    nodes=nodes,
                                    gtype="r-regular",
                                    d=d,
                                    n_jobs=n_jobs,
                                    M=m,
                                    seed=seed))

  print()
  print("Calculating times for Erdos-Renyi graph...")
  temp = pd.DataFrame(time_complexity(functions=functions,
                                      nodes=nodes,
                                      gtype="erdos-renyi",
                                      p=p,
                                      n_jobs=n_jobs,
                                      M=m,
                                      seed=seed))
  
  # Get an unique melted dataframe
  df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
  df = df.melt(['nodes', 'type'], var_name='cols', value_name='vals')

  # Save the dataframe if asked
  if save: df.to_parquet("time_complexity.parquet")

  # Plotting the time curves in normal scale...
  relplot = sns.relplot(data=df, x="nodes", y="vals", hue="cols",
                        col="type", kind='line')
  relplot.set(ylabel='Time', xlabel='Number of nodes', ylim=(-0.01, 1e-01))
  relplot._legend.set_title("Algorithms")
  relplot.fig.subplots_adjust(top=.85)
  relplot.fig.suptitle("Time Complexity curves in normal scale")
  relplot.fig.savefig("normal_scale.png")

  # Plotting the time curves in log scale
  relplot = sns.relplot(data=df, x="nodes", y="vals", hue="cols",
                        col="type", kind='line')
  relplot.set(ylabel='Log(Time)', xlabel='Number of nodes', yscale='log')
  relplot._legend.set_title("Algorithms")
  relplot.fig.subplots_adjust(top=.85)
  relplot.fig.suptitle("Time Complexity curves in log scale")
  relplot.fig.savefig("log_scale.png")

  # Add the log scaled values
  df["logtime"] = df.vals.apply(lambda x: log(x+1))
  df["lognodes"] = df.nodes.apply(lambda x: log(x+1))

  # Do the regression of the log scaled data
  regline = {}
  for t in df["type"].unique().tolist():
    regline[t] = {}
    for alg in df["cols"].unique().tolist():
      temp = df.loc[(df["type"] == t)&(df["cols"] == alg)]

      slope, intercept, r_value, p_value, std_err = linregress(temp['lognodes'],
                                                               temp['logtime'])
      regline[t][alg] = {'slope': round(slope, 2),
                         'intercept': round(intercept, 2),
                         'r_value': round(r_value, 2),
                         'p_value': round(p_value, 2),
                         'std_err': round(std_err, 2),
                         'ci': (round(slope-1.96*std_err, 2),
                                round(slope+1.96*std_err, 2))}

  # Plotting the time curves in log-log scale
  
  relplot = sns.FacetGrid(df, col="type",  hue="cols", height=5)
  relplot.map(sns.lineplot, "nodes", "vals")
  relplot.set(ylabel='Log(Time)', xlabel='Log(Number of nodes)', 
                yscale='log', xscale="log")
  relplot.fig.subplots_adjust(top=.80)
  relplot.fig.suptitle("Time Complexity curves in loglog scale")
  relplot.add_legend()
  relplot._legend.set_title("Algorithms")

  # Add a custom legend for each col plot with the slope
  # and the ci of each regline
  for i, t in enumerate(df["type"].unique().tolist()):
    axes = relplot.axes[0, i]
    axes.legend()
    for label in axes.get_legend().get_texts():

      slope = regline[t][label.get_text()]["slope"]
      ci = regline[t][label.get_text()]["ci"]

      label.set_text(f"Slope: {slope}, ci: {ci}")

  relplot.fig.savefig("loglog_scale.png")

  return None



def mc_sim_erdos_renyi(M: int = 10000,
                       nodes: int = 100,
                       n_jobs: int = -1,
                       save: bool = False,
                       seed: int = None) -> None:
  """A function to plot the probability of connectivity of a Erdos-Renyi graph.
  It saves the plot in the current directory as a 'prod_connectivity_erdos_renyi.png' file.

  Args:
    - M (int): Size of the simulation. Default 10000.
    - nodes (int): The number of nodes. Default 100.
    - n_jobs (int): The number of jobs for time_complexity, it defines the
                    number of cores to use, if -1 it all available cores are used.
                    Default -1.
    - save (bool): Save the measures dataframe in parquet format. Default False.
                   The dataframe will be saved as 'prob_connection_erdos_renyi.parquet'
                   in the current directory.
    - seed (int): Indicator of random number generation state. Default None.

  Returns:
    -  None
  """
  def simulate(nodes: int,
               p: float):
    return check_with_bfs(erdos_renyi_graph(n=nodes, p=p))

  # Set the configuration of Seaborn to the default one.
  sns.set()

  # Setting the random seed
  random.seed(seed)

  # The list of probabilities on which we simulate
  probs = list(np.arange(0, 1+0.01, 0.01))

  print()
  print("Running the simulation...")
  temp = {'p_c': []}
  for _ in tqdm(range(M)):
    temp['p_c'] += Parallel(n_jobs=n_jobs)(delayed(simulate)(nodes=nodes, p=p) for p in probs)
  
  temp['p'] = probs*M

  # DataFrame to use for the seaborn plot
  df = pd.DataFrame(temp)

  # Save the data is save == True
  if save: df.to_parquet("prob_connection_erdos_renyi.parquet")

  # Plot with seaborn
  print()
  print("Generating the plot...")
  plot = sns.relplot(data=df, x="p", y="p_c", kind='line')
  plot.set(title=f'Probability of Connectivity as function of\nthe probability of an edge to exist',
           ylabel="Probability of Connectivity",
           xlabel="Probability of an edge to exist")
  plt.axvline(x=log(nodes)/nodes, linestyle="dashed", alpha=0.3, color="red")
  plt.text(x=0, y=0.4, s="Point of Transition p=log(n)/n",
           color="red", rotation= 90)
  plot.fig.savefig("prob_connectivity_erdos_renyi.png", bbox_inches="tight")

  return None



def mc_sim_r_regular(M: int = 10000,
                     K: int = 100,
                     D: Iterable[int] = (2, 8),
                     n_jobs: int = -1,
                     save: bool = False,
                     seed: int = None) -> None:
  """A function to plot the probability of connectivity of a R-Regular graph.
  It saves the plot in the current directory as a
  prod_connectivity_r_regular.png' file.

  Args:
    - M (int): The size of the simulation. Default 10000.
    - K (int): The number of nodes. Default 100.
    - D (Iterable[int]): The iterable containing the the single d to check. Default (2, 8).
    - n_jobs (int): The number of jobs for time_complexity, it defines the
                    number of cores to use, if -1 it all available cores are used.
                    Default -1.
    - save (bool): Save the measures dataframe in parquet format. Default False.
                   The dataframe will be saved as 'prob_connection_erdos_renyi.parquet'
                   in the current directory.
    - seed (bool): Indicator of random number generation state. Default None. 

  Returns:
    - None
  """
  def simulate(nodes: int,
               d: int) -> Union[int, bool]:

    # Sanity check for Random Regular Graphs
    if not 0 <= d < nodes or (nodes * d) % 2 != 0:
        return None
    
    return check_with_bfs(nx.random_regular_graph(n=nodes, d=d))

  # Set the configuration of Seaborn to the default one.
  sns.set()

  # Setting the random seed
  random.seed(seed)

  # The final DataFrame for seabornb
  df = pd.DataFrame()

  # List of nodes
  nodes = list(range(K+1))

  for d in D:
    print()
    print(f"Running simulation for d={d}:")
    temp = {'p_c': []}
    for _ in tqdm(range(M)):
      temp['p_c'] += Parallel(n_jobs=n_jobs)(delayed(simulate)(nodes=k, d=d) for k in nodes)

    temp['K'] = nodes*M
    temp['r'] = d

    df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
    del temp

  df.dropna(inplace=True)

  if save: df.to_parquet("prob_connectivity_r_regular.parquet")

  print()
  print("Generating the plot...")
  # Plot with seaborn
  plot = sns.relplot(data=df, x="K", y="p_c", kind='line', hue='r',
                     palette=['r', 'b'])
  plot.set(title=f'Probability of Connectivity as function of K',
           ylabel="Probability of Connectivity")
  plot.fig.savefig("prob_connectivity_r_regular.png", bbox_inches="tight")

  return None





#############################################################
#                                                           #
#                 Second assignment stuff                   #
#                                                           #
#############################################################



class FatTree(nx.Graph):
  """A class that implements the Fat-Tree Topology for DC networks.

  Args:
    - n (int): The number of ports for each switches. Default 64.

  Attributes:
    - ports (int): Where we store the variable 'n', number of ports for each
                   switches.
    - pods (int): Number of pods, this is equal to n.
    - maximum_n_servers (int): Maximum number of servers, this is equal to int( (n**3)/4 )
    - maximum_n_switches (int): Maximum number of switches, this is equal to int( (5*n**2)/4 )
    - number_core_switches (int): Number of core switches, this is equal to int( (n/2)**2 ).
    - number_aggr_switches (int): Number of aggregation switches, this is equal to int( n**2/2 ).
    - number_edge_switches (int): Number of edge switches, this is equal to int( n**2/2 ).
    - number_servers_per_pod (int): Number of servers in each pod, this is equal to int( (n/2)**2 )
    - servers (List[int]): The list of nodes that are servers.
    - core_switches (List[int]): The list of nodes that are core switches.
    - aggr_switches (List[int]): The list of nodes that are aggregation switches.
    - edge_switches (List[int]): The list of nodes that are edge switches.
    - servers_closeness (Dict[int, Tuple[Dict[int, int], Dict[int, List[int]]]]): The dict in which we save for each node its node_tree and distances with the other nodes.
  """
  def __init__(self,
               n: int = 64) -> None:
    super().__init__()
    self.__n = self.__pods = n
    self.__maximum_n_servers = int( (n**3)/4 )
    self.__maximum_n_switches = int( (5*n**2)/4 )
    self.__number_core_switches = self.__number_servers_per_pod = int( (n/2)**2 )
    self.__number_aggr_switches = self.__number_edge_switches = int( n**2/2 )
    self.__servers_closeness = {}
    
    # Create the core switches
    self.__core_switches = list(range(self.number_core_switches))
    self.add_nodes_from(self.__core_switches, type="core", pod=None)


    for pod in range(self.pods):

      # For each pod we create the aggregation switches
      aggr_switches = []
      for i, aggr_switch in enumerate(range(self.number_of_nodes(), self.number_of_nodes()+int(self.ports/2))):
        self.add_node(aggr_switch, type='aggr', pod=pod)
        aggr_switches.append(aggr_switch)

        # We connect each aggregation switch to its n/2 core switches
        for core_switch_index in range(i*(int(n/2)), int(n/2)+i*(int(n/2))):
          self.add_edge(aggr_switch, self.core_switches[core_switch_index], pod=pod)

      # For each pod we create the edge switches
      edge_switches = []
      for edge_switch in range(self.number_of_nodes(), self.number_of_nodes()+int(self.ports/2)):
        self.add_node(edge_switch, type='edge', pod=pod)
        edge_switches.append(edge_switch)

        # We connect each edge switch to each aggregation switch of the same pod
        for aggr_switch in aggr_switches:
          self.add_edge(edge_switch, aggr_switch, pod=pod)

      for edge_switch in edge_switches:
        # For each edge switch we create and connect n/2 servers
        for server in range(self.number_of_nodes(), self.number_of_nodes()+int(self.ports/2)):
          self.add_node(server, type='server', pod=pod)
          self.add_edge(edge_switch, server, pod=pod)

    self.__servers = [node for node, attr in self.nodes(data=True) if attr['type']=='server']
    self.__aggr_switches = [node for node, attr in self.nodes(data=True) if attr['type']=='aggr']
    self.__edge_switches = [node for node, attr in self.nodes(data=True) if attr['type']=='edge']


  #############################
  #   Functions properties    #
  #############################

  @property
  def ports(self) -> int:
    return self.__n
  
  @property
  def pods(self) -> int:
    return self.__pods

  @property
  def maximum_n_servers(self) -> int:
    return self.__maximum_n_servers

  @property
  def maximum_n_switches(self) -> int:
    return self.__maximum_n_switches

  @property
  def number_core_switches(self) -> int:
    return self.__number_core_switches

  @property
  def number_servers_per_pod(self) -> int:
    return self.__number_servers_per_pod

  @property
  def number_aggr_switches(self) -> int:
    return self.__number_aggr_switches

  @property
  def number_edge_switches(self) -> int:
    return self.__number_edge_switches

  @property
  def servers(self) -> List[int]:
    return self.__servers

  @property
  def edge_switches(self) -> List[int]:
    return self.__edge_switches

  @property
  def aggr_switches(self) -> List[int]:
    return self.__aggr_switches

  @property
  def core_switches(self) -> List[int]:
    return self.__core_switches

  @property
  def servers_closeness(self) -> Dict[int, Tuple[Dict[int, int], Dict[int, List[int]]]]:
    return self.__servers_closeness

  #############################
  #   Functions definitions   #
  #############################


  def summary(self) -> None:
    """Return a summary of the Fat-Tree graph.

    Returns:
      - str : The summary string.
    """
    summary=f"""
    Graph summary:
    \tNumber of nodes: {self.number_of_nodes()}
    \tNumber of edges: {self.number_of_edges()}

    Fat-Tree parameters summary: 
    \tNumber of ports per switch: {self.ports}
    \tNumber of pods: {self.pods}
    \tNumber of servers per pod: {self.number_servers_per_pod}
    \tNumber of total servers: {self.maximum_n_servers}
    \tNumber of total switches: {self.maximum_n_switches}
    \t\t- Number of core switches: {self.number_core_switches}
    \t\t- Number of aggregation switches: {self.number_aggr_switches}
    \t\t- Number of edge switches: {self.number_edge_switches}
    """
    print(summary)
    return None


  def draw(self,
           prog: str = 'dot') -> None:
    """Draw the graph using graphviz_layout.

    Args:
      - prog (str): The prog variable for the graphviz_layout. Default 'dot'.

    Returns:
      - None
    """
    sns.set_theme(style='white')
    pos = graphviz_layout(self, prog=prog)
    nx.draw_networkx_edges(self, pos)
    nx.draw_networkx_nodes(self, pos, nodelist=self.core_switches, node_color='green')
    nx.draw_networkx_nodes(self, pos, nodelist=self.aggr_switches, node_color='yellow')
    nx.draw_networkx_nodes(self, pos, nodelist=self.edge_switches, node_color='red')
    nx.draw_networkx_nodes(self, pos, nodelist=self.servers, node_color='blue')
    plt.title(f"Fat-Tree Topology\n{self.ports} ports.")
    plt.show()
    return None


  def is_server(self,
                node: int) -> bool:
    """Checks if the passed node is a server or not.
    
    Args:
      - node (int): The id of the node.
    
    Returns:
      - bool : True if it is a server, else False.
    """
    return bisect_left(self.servers, node) <  self.maximum_n_servers


  def reset_servers_closeness(self) -> None:
    """Reset the servers_closeness dictionary to an empty dictionary.

    Return:
      - None
    """
    self.__servers_closeness = {}
    return None


  def get_closest_servers(self,
                          node: int,
                          n: int,
                          hopes: bool = False,
                          verbose: bool = True,
                          seed: int = None) -> Union[List[int], Tuple[List[int], List[int]]]:
    """Return a list with the 'n' closest servers to the passed node server.

    Args:
      - node (int): The chosen starting node.
      - n (int): The number of closest servers.
      - hopes (bool): If True the hopes are also returned, if False only the servers. Default False.
      - verbose (bool): If True log info are displayed. Default True.
      - seed (int): Indicator of random number generation state. Default None.

    Returns:
      - final (Union[List[int], Tuple[List[int], List[int]]]): If hopes is set to False then the list with the closest
                                                               servers to the passed node is returned, otherwise it is
                                                               returned a tuple of the list of the closest servers and
                                                               their number of hopes.
    """
    assert self.has_node(node), "The passed node is not in the graph!"
    assert n <= self.maximum_n_servers, f"The maximum number of servers is {self.maximum_n_servers}, but passed {n}."

    # Set random seed
    if verbose and seed: print(f"Setting the seed to {seed}", end="")
    random.seed(seed)
    if verbose and seed: print(f", done.\n")

    if node not in self.servers_closeness:
      # Getting node_tree for the passed node
      if verbose: print(f"Getting the path tree with root the passed node {node}", end="")
      node_tree = nx.single_source_shortest_path_length(self, node)
      if verbose: print(", done.\n")

      # Creating the distance dictionary
      if verbose: print("Creating the distance dictionary", end="")
      distances = {}
      for server in node_tree:
        if self.is_server(server) and server != node:
          if distances.get(node_tree[server]):
            distances[node_tree[server]].append(server)
          else:
            distances[node_tree[server]] = [server]
      if verbose: print(", done.\n")
      self.servers_closeness[node] = (node_tree, distances)
    else:
      node_tree, distances = self.servers_closeness[node]

    # Evaluating
    if verbose: print("Evaluating the results", end="")
    results = []
    for servers in distances.values():
      if n == 0: break
      if len(servers) <= n:
        results += servers
        n -= len(servers)
      else:
        results += random.choices(servers, k=n)
        n = 0
    if verbose: print(", done.\n")

    # Branching for hopes
    if hopes:
      final = (results, [node_tree[server] for server in results])
    else:
      final = results

    return final



class Jellyfish(nx.Graph):
  """A class that implements the Jellyfish topology for Data Centers.

  Args:
    - nodes (int): The number of switches present in the DC.
    - n (int): The number of the switches ports. Default 64.

  Attributes:
    - ports (int): The variable where it is stored the value of argument n.
                   The number of ports for each switch.
    - number_of_switches (int): The number of switches, it is equal to the
                                passed variable 'node'
    - number_of_servers (int): The number of servers, it is equal to the
                               number_of_switches * int( self.ports/2 )
    - servers (List[int]): The list of servers in the DC.
    - switches (List[int]): The list of switches in the DC.
    - servers_closeness (Dict[int, Tuple[Dict[int, int], Dict[int, List[int]]]]): The dict in which we save for each node its node_tree and distances with the other nodes.
  """
  def __init__(self,
               nodes: int,
               n: int = 64) -> None:
    super().__init__()
    self.__n = n
    self.__number_of_switches = nodes
    self.__number_of_servers = self.number_of_switches * int( self.ports/2 )
    self.__servers_closeness = {}

    # Add the R-Regular structure between switches
    self.add_edges_from(list(map(lambda x: [int(node) for node in x.split(" ")],
                                 nx.generate_edgelist(nx.random_regular_graph(n=self.number_of_switches, d=int(self.ports/2)), data=False))))

    # Set the attrubutes for the switches: node and edges attributes
    nx.set_node_attributes(self, "switch", name="type")
    nx.set_edge_attributes(self, "with_switch", name="type")

    # Add the servers
    nodes = list(self.nodes())
    for node in nodes:
      for server in range(self.number_of_nodes(), self.number_of_nodes() + int(self.ports/2)):
        self.add_node(server, type='server')
        self.add_edge(node, server, type="with_server")

    self.__servers = [node for node, attr in self.nodes(data=True) if attr['type'] == 'server']
    self.__switches = [node for node, attr in self.nodes(data=True) if attr['type'] == 'switch']

  
  
  #############################
  #   Functions properties    #
  #############################

  @property
  def ports(self) -> int:
    return self.__n

  @property
  def number_of_switches(self) -> int:
    return self.__number_of_switches

  @property
  def number_of_servers(self) -> int:
    return self.__number_of_servers

  @property
  def servers(self) -> List[int]:
    return self.__servers

  @property
  def switches(self) -> List[int]:
    return self.__switches

  @property
  def servers_closeness(self) -> Dict[int, Tuple[Dict[int, int], Dict[int, List[int]]]]:
    return self.__servers_closeness

  #############################
  #   Functions definitions   #
  #############################

  def summary(self) -> None:
    """Return a summary of the Jellyfish graph.

    Returns:
      - str : The summary string.
    """
    summary=f"""
    Graph summary:
    \tNumber of nodes: {self.number_of_nodes()}
    \tNumber of edges: {self.number_of_edges()}

    Jellyfish parameters summary: 
    \tNumber of ports per switch: {self.ports}
    \tNumber of total servers: {self.number_of_servers}
    \tNumber of total switches: {self.number_of_switches}
    """
    print(summary)
    return None


  def draw(self) -> None:
    """Draw the graph using nx.kamada_kawai_layout().

    Returns:
      - None
    """
    pos = nx.kamada_kawai_layout(self)
    nx.draw_networkx_edges(self, pos)
    nx.draw_networkx_nodes(self, pos, nodelist=self.servers, node_color='blue', node_size=150)
    nx.draw_networkx_nodes(self, pos, nodelist=self.switches, node_color='red')
    plt.title(f"Jellyfish Topology\n{self.number_of_switches} switches and {self.ports} ports.")
    plt.show()
    return None


  def is_server(self,
                node: int) -> bool:
    """Checks if the passed node is a server or not.
    
    Args:
      - node (int): The id of the node.
    
    Returns:
      - bool : True if it is a server, else False.
    """
    return bisect_left(self.servers, node) < self.number_of_servers

  
  def reset_servers_closeness(self) -> None:
    """Reset the servers_closeness dictionary to an empty dictionary.

    Return:
      - None
    """
    self.__servers_closeness = {}
    return None


  def get_closest_servers(self,
                          node: int,
                          n: int,
                          hopes: bool = False,
                          verbose: bool = True,
                          seed: int = None) -> Union[List[int], Tuple[List[int], List[int]]]:
    """Return a list with the 'n' closest servers to the passed node server.

    Args:
      - node (int): The chosen starting node.
      - n (int): The number of closest servers.
      - hopes (bool): If True the hopes are also returned, if False only the servers. Default False.
      - verbose (bool): If True log info are displayed. Default True.
      - seed (int): Indicator of random number generation state. Default None.

    Returns:
      - final (Union[List[int], Tuple[List[int], List[int]]]): If hopes is set to False then the list with the closest
                                                               servers to the passed node is returned, otherwise it is
                                                               returned a tuple of the list of the closest servers and
                                                               their number of hopes.
    """
    assert self.has_node(node), "The passed node is not in the graph!"
    assert n <= self.number_of_servers, f"The maximum number of servers is {self.number_of_servers}, but passed {n}."

    # Set random seed
    if verbose and seed: print(f"Setting the seed to {seed}", end="")
    random.seed(seed)
    if verbose and seed: print(f", done.\n")

    if node not in self.servers_closeness:
      # Getting node_tree for the passed node
      if verbose: print(f"Getting the path tree with root the passed node {node}", end="")
      node_tree = nx.single_source_shortest_path_length(self, node)
      if verbose: print(", done.\n")

      # Creating the distance dictionary
      if verbose: print("Creating the distance dictionary", end="")
      distances = {}
      for server in node_tree:
        if self.is_server(server) and server != node:
          if distances.get(node_tree[server]):
            distances[node_tree[server]].append(server)
          else:
            distances[node_tree[server]] = [server]
      if verbose: print(", done.\n")
      self.servers_closeness[node] = (node_tree, distances)
    else:
      node_tree, distances = self.servers_closeness[node]

    # Evaluating
    if verbose: print("Evaluating the results", end="")
    results = []
    for servers in distances.values():
      if n == 0: break
      if len(servers) <= n:
        results += servers
        n -= len(servers)
      else:
        results += random.choices(servers, k=n)
        n = 0
    if verbose: print(", done.\n")

    # Branching for hopes
    if hopes:
      final = (results, [node_tree[server] for server in results])
    else:
      final = results

    return final  



def average_throughput(times: np.array,
                       C: float) -> np.array:
  """This function returns a np.array of values for the throughput given a
  np.array of RTT times and the C capacity of each link of the DC network.

  Args:
    - times (np.array): An np.array containing the RTT times.
    - C (float): The capacity of each link of the DC network.
  
  Returns:
    - np.array : The throughput np.array.
  """
  return (C*(1/times))/np.mean(times)



def response_times(hopes: np.array,
                   Lf: float,
                   Loi: np.array,
                   C: int,
                   f: float,
                   tau: float,
                   T: int,
                   Xi: np.array) -> np.array:
  """The function calculates the response time for each connection.

  Args:
    - hopes (np.array): The np.array with all the hopes for each connection.
    - Lf (float): The input file length, it should be divided first by the
                  number of connections.
    - Loi (np.array): The output file length of each connection.
    - C (int): The bandwidth of the DC links.
    - f (float): The proportion of the header for the TCP connection. 
    - tau (float): The tau parameter for calculating the Ti times.
    - T (int): The costant task time to add.
    - Xi (np.array): The variables component for the time used to run the task
                     on each distribuited server.

  Returns:
    - np.array : The response time for each connection.
  """
  Ti = 2*tau*hopes

  time_in_going =  ( Lf*(1+f) ) / average_throughput(Ti, C)
  time_in_returning = ( Loi*(1+f) ) / average_throughput(Ti, C)
  working_time = T + Xi

  return time_in_going + working_time + time_in_returning



def simulation_response_time(C: int = 10e09, #bit/s
                             tau: float = 5e-06, #s
                             Lf: int = 4e12, #B
                             Lo: int = 4e12, #B
                             Ex: int = 8*60*60, #s
                             T: int = 30, #s
                             f: float = 48/1500,
                             n: int = 64,
                             start: int = 1,
                             n_jobs: int = -1,
                             M: int = 1000,
                             N: int = 10000,
                             save: bool = False,
                             baseline: bool = True,
                             seed: int = None) -> None:
  """The function used to simulate the response time.

  Args:
    - hopes (np.array): The np.array with all the hopes for each connection.
    - C (int): The bandwidth of the DC links in bit/s. Default 10e09.
    - tau (float): The tau parameter for calculating the Ti times in seconds.
                   Default 5e-06.
    - Lf (int): The input file length in Bytes. Default 4e12.
    - Lo (int): The total output file length in Bytes. Default 4e12.
    - Ex (int): The average for the variable component of the time
                used to run the task locally, in seconds. Default 8*60*60.
    - T (int): The costant task time to add in seconds. Default 30.
    - f (float): The proportion of the header for the TCP connection.
                 Default 48/1500.
    - n (int): The number of switch ports. Default 64.
    - start (int): The node to start from. Default 1.
    - n_jobs (int): The number of jobs for time_complexity, it defines the
                    number of cores to use, if -1 it all available cores are
                    used. Default -1.
    - M (int): The size of the simulation. Default 1000.
    - N (int): The number of servers to use. Default 10000.
    - save (bool): If True save the results as a parquet file called 'response_time.parquet',
                   else False. Default False.
    - baseline (bool): If True simulate and plot also the baseline, else False. Default True.
    - seed (int): Indicator of random number generation state. Default None.

  Returns:
    - None
  """
  def simulate(n: int,
               A: int,
               g: nx.Graph) -> float:
    servers, hopes = g.get_closest_servers(node=A, n=n, hopes=True, verbose=False)

    Xi = np.random.exponential(scale=(Ex/n), size=n)
    Loi = np.random.uniform(low=0, high=(2*Lo)/n, size=n)

    response_time = np.max(response_times(np.array(hopes), Lf=Lf/n,
                                          Loi=Loi, C=C, f=f, tau=tau, T=T, Xi=Xi) / (T+Ex))

    return response_time

  assert start > 0, "The number of closest servers can be only greater than 0"

  # Set the default configuration for seaborn
  sns.set()

  # Set the seed
  random.seed(seed)

  # Initialize the step, nodes and the df
  step = 10**(int(log10(N))-2)
  nodes = list(range(start, N+step, step))
  df = pd.DataFrame()

  if baseline:
    print("Simulation for the Baseline...")
    for _ in range(M):
      temp = pd.DataFrame()
      temp['N'] = nodes
      temp['graph'] = 'Baseline'
      temp['response_time'] = (T+np.random.exponential(scale=Ex, size=len(nodes))) / (T+Ex)

      df = pd.concat([df, temp], ignore_index=True)
    del temp

  print()
  print("Simulation for the Fat-Tree topology...")
  # Due to the fact that the Fat-Tree topology is deterministic it is possible to initialize the graph
  # outside the simulation for, and due to the fact that the structure is regular the choice of the
  # random server A doesn't affect the simulation results, for this we initialize A outside the simulation.
  g = FatTree(n=n)
  A = random.choice(g.servers)
  temp = {'response_time': []}
  for _ in tqdm(range(M)):
    temp['response_time'] += Parallel(n_jobs=n_jobs)(delayed(simulate)(n=i, A=A, g=g) for i in nodes)
      
  temp['N'] = nodes * M
  temp['graph'] = 'Fat-Tree'

  df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
  del temp

  switches = int((2*g.maximum_n_servers)/n)

  print()
  print("Simulation for the Jellyfish topology...")
  temp = {'response_time': []}
  for _ in tqdm(range(M)):
    g = Jellyfish(nodes=switches, n=n)
    A = random.choice(g.servers)
    temp['response_time'] += Parallel(n_jobs=n_jobs)(delayed(simulate)(n=i, A=A, g=g) for i in nodes)
      
  temp['N'] = nodes * M
  temp['graph'] = 'Jellyfish'

  df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
  del temp

  if save: df.to_parquet('response_time.parquet')

  print()
  print("Creating the plots", end="")
  # Plot with seaborn
  plot = sns.relplot(data=df, x="N", y="response_time", kind="line", hue='graph')
  ylim = (0, 1.25) if baseline else (0, 0.005)
  plot.set(title=f'Response Time curve', ylim=ylim,
           ylabel="Response Time")
  plot.fig.savefig("response_time.png", bbox_inches="tight")

  # Plot with seaborn
  plot = sns.relplot(data=df, x="N", y="response_time", kind="line", hue='graph')
  plot.set(title=f'Response Time curve in loglog scale',
           ylabel="Log(Response Time)", xlabel="Log(N)",
           xscale='log', yscale='log')
  plot.fig.savefig("response_time_loglog.png", bbox_inches="tight")
  print(", done.")

  return None



def simulation_job_running_cost(C: int = 10e09, #bit/s
                                tau: float = 5e-06, #s
                                Lf: int = 4e12, #B
                                Lo: int = 4e12, #B
                                Ex: int = 8*60*60, #s
                                T: int = 30, #s
                                eps: float = 0.1,
                                f: float = 48/1500,
                                n: int = 64,
                                start: int = 1,
                                n_jobs: int = -1,
                                M: int = 1000,
                                N: int = 10000,
                                save: bool = False,
                                baseline: bool = True,
                                seed: int = None):
  """The function used to simulate the job running cost.

  Args:
    - hopes (np.array): The np.array with all the hopes for each connection.
    - C (int): The bandwidth of the DC links in bit/s. Default 10e09.
    - tau (float): The tau parameter for calculating the Ti times in seconds.
                   Default 5e-06.
    - Lf (int): The input file length in Bytes. Default 4e12.
    - Lo (int): The total output file length in Bytes. Default 4e12.
    - Ex (int): The average for the variable component of the time
                used to run the task locally, in seconds. Default 8*60*60.
    - T (int): The costant task time to add in seconds. Default 30.
    - eps (int): The value for eps. Default 0.1.
    - f (float): The proportion of the header for the TCP connection.
                 Default 48/1500.
    - n (int): The number of switch ports. Default 64.
    - start (int): The node to start from. Default 1.
    - n_jobs (int): The number of jobs for time_complexity, it defines the
                    number of cores to use, if -1 it all available cores are
                    used. Default -1.
    - M (int): The size of the simulation. Default 1000.
    - N (int): The number of servers to use. Default 10000.
    - save (bool): If True save the results in a parquet file called 'job_running_cost.parquet'.
                   Default False.
    - baseline (bool): If True simulate and plot also the baseline. Default True.
    - seed (int): Indicator of random number generation state. Default None.

  Returns:
    - None
  """
  def simulate(n: int,
               A: int,
               g: nx.Graph,
               expected_s_baseline: float) -> float:
    servers, hopes = g.get_closest_servers(node=A, n=n, hopes=True, verbose=False)

    Xi = np.random.exponential(scale=(Ex/n), size=n)
    Loi = np.random.uniform(low=0, high=(2*Lo)/n, size=n)

    response_time = np.max(response_times(np.array(hopes), Lf=Lf/n, Loi=Loi,
                                          C=C, f=f, tau=tau, T=T, Xi=Xi))
    
    servers_usage = T + Xi

    job_running_cost = response_time + eps*np.sum(servers_usage)

    return job_running_cost / expected_s_baseline

  assert start > 0, "Servers can be only from 1 on"

  # Set the default config for seaborn
  sns.set()

  # Set the random seed
  random.seed(seed)

  # Set the step, nodes and df
  step = 10**(int(log10(N))-2)
  nodes = list(range(start, N+step, step))
  df = pd.DataFrame()

  # Save the expected S_baseline value for the normalization
  expected_s_baseline = (T+Ex) + eps*(T+Ex)

  if baseline:
    print("Simulation for the Baseline...")
    for _ in range(M):
      temp = pd.DataFrame()
      temp['N'] = nodes
      temp['graph'] = 'Baseline'

      response_time = T+np.random.exponential(scale=Ex, size=len(nodes))

      server_usage = response_time

      job_running_cost = response_time + eps*server_usage

      temp['job_running_cost'] = job_running_cost / expected_s_baseline

      df = pd.concat([df, temp], ignore_index=True)
    del temp

  print()
  print("Simulation for the Fat-Tree topology...")
  # Due to the fact that the Fat-Tree topology is deterministic it is possible to initialize the graph
  # outside the simulation for, and due to the fact that the structure is regular the choice of the
  # random server A doesn't affect the simulation results, for this we initialize A outside the simulation.
  g = FatTree(n=n)
  A = random.choice(g.servers)
  temp = {'job_running_cost': []}
  for _ in tqdm(range(M)):
    temp['job_running_cost'] += Parallel(n_jobs=n_jobs)(delayed(simulate)(n=i, A=A, g=g, expected_s_baseline=expected_s_baseline) for i in nodes)

  temp['N'] = nodes * M
  temp['graph'] = 'Fat-Tree'

  df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
  del temp

  switches = int((2*g.maximum_n_servers)/n)

  print()
  print("Simulation for the Jellyfish topology...")
  temp = {'job_running_cost': []}
  for _ in tqdm(range(M)):
    g = Jellyfish(nodes=switches, n=n)
    A = random.choice(g.servers)
    temp['job_running_cost'] += Parallel(n_jobs=n_jobs)(delayed(simulate)(n=i, A=A, g=g, expected_s_baseline=expected_s_baseline) for i in nodes)

  temp['N'] = nodes * M
  temp['graph'] = 'Jellyfish'

  df = pd.concat([df, pd.DataFrame(temp)], ignore_index=True)
  del temp

  if save: df.to_parquet('job_running_cost.parquet')

  print()
  print("Finding, via bootstrap, the number of servers required for the minimum job running cost beside the Baseline...")
  minimums = {}
  # Get the values and nodes of the minimum for FatTree and Jellyfish
  for gtype in tqdm(df.graph.unique().tolist()):

    # Jump the Baseline
    if gtype == "Baseline" : continue
    
    temp = {}
    for n in nodes:
      # Take from the df the multiple simulations for each node n in nodes
      # And set in the right format for performing the bootstrap estimation
      data = (df[(df.N == n)&(df.graph == gtype)].job_running_cost.to_numpy(), )
      temp[n] = [np.mean(bootstrap(data, np.mean, random_state=seed).bootstrap_distribution)]

    minimums[gtype] = (min(temp, key=temp.get), round(min(temp.values())[0], 4))

  # Printing the results for the finded minimums
  print()
  for gtype in minimums:
    idx, value = minimums[gtype]
    print(f"For the {gtype} topology the minimum {value} is with {idx} servers")
    print()

  print("Creating the plots", end="")
  # Plot with seaborn
  plot = sns.relplot(data=df, x="N", y="job_running_cost", kind="line", hue='graph')
  plot.set(title=f'Job Running Cost curve',# ylim=(0, 1.25),
           ylabel="Job Running Cost")
  plot.fig.savefig("job_running_cost.png", bbox_inches="tight")

  # Plot with seaborn
  plot = sns.relplot(data=df, x="N", y="job_running_cost", kind="line", hue='graph')
  plot.set(title=f'Job Running Cost curve in loglog scale',
           ylabel="Log(Job Running Cost)", xlabel="Log(N)",
           xscale='log', yscale='log')
  plot.fig.savefig("job_running_cost_loglog.png", bbox_inches="tight")
  print(", done.")

  return None






if __name__ == "__main__":
    import argparse

    description = """
    In this python file there are all the classes and function needed to solve the NBD Challenge 1.
    
    It is possible to run this file as a python script with the pip module 'memory-profiler' to
    check the memory usage for the following methods for checking a graph connectivity:
    - Bread First Search
    - Laplacian
    - Irreducibility
  
    This script is meant to be used with the mprof executable.
    To run you have to install 'memory-profiler', to do so 'pip install memory-profiler'.
  
    Now you can use the mprof executable on this script, as:
    'mprof run script.py'

    To check the available parameters just run 'python script.py -h'.
    """

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
  
    parser.add_argument('-n',
                        '--nodes',
                        help='The number of nodes for the graph. Default 100.',
                        default=10000,
                        type=int)
  
    parser.add_argument('-g',
                        '--graph',
                        help='Type of graph.\nDefault value is "r-regular".\nPossible values are:\n- "erdos-renyi" : For a Erdos-Renyi graph.\n- "r-regular" : For a R-Regular graph.',
                        default='r-regular',
                        type=str)
  
    parser.add_argument('-p',
                        '--probability',
                        help='The probability for the Erdos-Renyi graph. Defaul is 0.01',
                        default=0.01,
                        type=float)
  
    parser.add_argument('-d',
                        '--degree',
                        help='The degree parameter for the D-Regular or R-Regular graph. Dafault is 3.',
                        default=3,
                        type=int)
  
    parser.add_argument('-a',
                        '--algorithm',
                        help='The algorithm to check the connectivity.\nDefault value is "bfs".\nPossible values are:\n- "bfs" : For the Bread First Search one.\n- laplacian : For the Laplacian one.\n- irreducibility : For the Irreducibility one.',
                        default='bfs',
                        type=str)


    args = parser.parse_args()

  
    match args.graph:

        case "r-regular":
            print(f"Creating the graph of type {args.graph}", end="")
            g = nx.random_regular_graph(n=args.nodes, d=args.degree)

        case "erdos-renyi":
            print(f"Creating the graph of type {args.graph}", end="")
            g = erdos_renyi_graph(n=args.nodes, p=args.probability)

        case _:
            raise Exception(f"The 'graph' parameter can be only 'r-regular' or 'erdos-renyi', passed {args.graph}")

    print(", done.")

    match args.algorithm:
    
        case "bfs":
            print(f"Check the connectivity with {args.algorithm} algorithm", end="")
            check_with_bfs(g)

        case "laplacian":
            print(f"Check the connectivity with {args.algorithm} algorithm", end="")
            check_with_laplacian(g)

        case "irreducibility":
            print(f"Check the connectivity with {args.algorithm} algorithm", end="")
            check_with_irreducibility(g)

        case _:
            raise Exception(f"The 'algorithm' parameter can only be 'bfs', 'laplacian' or 'irreducibility', passed {args.algorithm}.")

    print(", done.")
