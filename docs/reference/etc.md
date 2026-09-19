# ETC API

Use `cubesim.Etc` to configure and execute a forward exposure-time
calculation. The same class is available from the package root:

```python
from cubesim import Etc
```

The object is mutable during configuration. Each call to `run()` returns an
independent immutable result. See the [Python API guide](../api.md) for the
complete lifecycle and [Results And Persistence](results.md) for returned
products.

::: cubesim.etc.Etc
    options:
      members_order: source
      show_root_heading: true
      show_source: false
