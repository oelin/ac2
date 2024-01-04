# AC2

A PettingZoo AEC environment for Ant Colony Coverage (AC2).

## Usage

```python
from ac2 import AC2, AC2Configuration

configuration = AC2Configuration(
    map_size=16,
    number_of_agents=16,
    number_of_obstacles=16,
    difficulty=0.01,
    duration=1000,
)

env = AC2(configuration=configuration)
```

The environment can be parallelized using `aec_to_parallel()`.

```python
from pettingzoo.utils.conversions import aec_to_parallel

parallel_env = aec_to_parallel(env)
```
