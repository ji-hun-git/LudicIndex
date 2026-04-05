# Surrogates

Place your production surrogate environment modules in this directory.

Each surrogate should expose a factory callable in `module:object` format, for example:

```python
from ludic_ai.env import POMDPEnv

class MySurrogate(POMDPEnv):
    ...


def build_env(**kwargs):
    return MySurrogate(**kwargs)
```

Then point `configs/production_surrogate_template.yaml` to:

```yaml
environment:
  factory: surrogates.my_surrogate:build_env
```

No other part of the pipeline needs to change.
