# vLLM + VPO-RM training image

This image is derived from the cluster-provided vLLM image and adds only the
two packages missing from that image for this repository: PEFT and PyArrow.

The final artifact belongs in the internal registry, not in `models/` or the
project's shared filesystem.  A suggested tag is:

```text
registry.h.pjlab.org.cn/ailab/vpo-rm-vllm-train:cu129-20260909
```

The exact repository path must be one to which the account has push access.
Build and push it from an authorized image-builder/development machine, then
use the pushed reference in `rjob submit --image`.
