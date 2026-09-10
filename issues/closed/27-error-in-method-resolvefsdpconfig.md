---
number: 27
title: "Error in method resolve_fsdp_config"
state: closed
labels:
- bug
---

In file [clm_training.fabric.distributed.py](https://github.com/gplsi/LMTK/blob/master/src/tasks/clm_training/fabric/distributed.py) the method `resolve_fsdp_config' is invoked passing the config as object with method `self.config.__dict__`.

```python
fsdp_config = resolve_fsdp_config(
                config=self.config.__dict__,
                model_name=self.config.model_name
            )
```

The problem is that then in that method  is related to the config object  being  dictionary and passed as an object that  only contains mappings of the attributes which causes loading only the default alue.

I suggest to add `self.config.to_dict()`