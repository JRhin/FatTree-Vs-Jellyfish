# Fat-tree Vs Jellyfish

This repository is the solution to the Challenge 1 of Networking for Big Data.

The challenge was about the analysis of connectivity in different types of graphs and implementing the Fat-tree and Jellyfish data center topologies (highlighting differences or similarities regarding *response time*).

## Repository Structure

```
./root
  |_ report.pdf
  |_ README.md
  |_ .gitignore
  |_ requirements.txt
  |_ 'Space Complexity images'/
  |   |_ 'bfs space complexity.png'
  |   |_ 'laplacian space complexity.png'
  |_ src/
      |_ script.py
```

## Dependencies

Having python (>= 3.10) and pip installed you can install (python enviroment is recommended) the requirements by running:

```bash
pip install -r requirements.txt
```
**Note**: `pygraphviz` needs `graphviz` to be installed on your system.

## How to run

### Run as a library

It is possible to import the `script.py` file and use the needed functions, example:

```python
# Import the file
from src import script

# Example of a function usage
script.simulation_response_time()
```

### Run as a script

It is possible to run the `script.py` as an executable in combination with `mprof` to get the space complexity of *BFS*, *Laplacian* and *Irreducibility*.

From a terminal shell, in the same directory of the script, run:

```bash
mprof run script.py
```

To show the plots then:

```bash
mprof plot
```
