# ReePy 

ReePy is a Reeb graph construction library containing accelerated algorithms for constructing Reeb graphs for multiple data modalities.

Currently supported classes include:

- A top level ReebGraph implementation for *any* data

- An implementation for trajectories of fixed length

- An implementation for trajectories with thresholded distances as a connectivity metric 

Planned features which still need to be integrated:

- A GPU accelerated implementation for graph analysis

- An implementation for mesh analysis

## Installation

You can install ReePy using uv from the GitHub. ReePy should work with other Python package managers as well. 

```
uv add git+https://github.com/UCSB-VRL/ReePy.git
```

## Philosophy

[Reeb graphs](https://en.wikipedia.org/wiki/Reeb_graph) capture the evolution of connected components over a series of level sets. For any Reeb graph in ReePy, we must define `level` which is a function that maps a point from the input modality to a number (either integer or floating point). To define what a connected component is, we must define `connectivity` which, given a collection of points, returns `True` if they are connected and `False` if they are not. Finally, we must define `continuity`, which given two points, returns `True` if one "evolves" from another. In the trajectory context, continuity would return `True` if the two points are consecutive in the same trajectory.

For example, for [Sequential Reeb Graphs](https://doi.org/10.1007/978-3-032-31583-0_15), you might use the following definitions:

```python
import itertools
import math
from reepy import ReebGraph

# each point p is (traj_index, sequence_index, x, y)
epsilon = 1 

level = lambda p: p[1]
connectivity = lambda pts: all(math.dist(p[2:], q[2:]) <= epsilon for p, q in itertools.combinations(pts, 2))
continuity = lambda p, q: p[0] == q[0] and abs(p[1] - q[1]) == 1

graph = ReebGraph(level=level, connectivity=connectivity, continuity=continuity)
```

Once a ReebGraph is defined, you can add data using the `ReebGraph.add(self, data)` method, which incrementally constructs the Reeb graph. You may observe that ReebGraph construction times scale exponentially with the number of points added, which may make using the base class impractical for your data. For this reason, we have modality-specific accelerated ReebGraph implementations, each of which make assumptions about the structure of the data. However, they still will define `connectivity`, `continuity` and `level` so an equivalent graph can be computed (and verified) against the base class.
