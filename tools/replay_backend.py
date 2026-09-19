"""Compile/load a trusted Tila backend failure artifact, without launching a kernel."""
import argparse
import json
import linecache
from pathlib import Path


def replay(folder):
    import torch
    import triton
    import triton.language as tl
    from tila import dtypes as D
    from tila.backend import preflight, failure
    from tila.target import resolve_cuda

    folder = Path(folder).resolve()
    context = json.loads((folder / "failure.json").read_text())
    if context["schema"] != "tila.backend-failure.v1":
        raise ValueError("unsupported backend artifact schema")
    target = resolve_cuda(torch, triton, torch.device("cuda", torch.cuda.current_device()))
    for key in ("torch_version", "cuda_version"):
        if getattr(target, key) != context["target"][key]:
            raise ValueError(f"replay environment mismatch: {key}")
    source = (folder / "kernel.py").read_text()
    filename = str(folder / "kernel.py")
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": "tila_backend_replay"}
    source_map = {int(k): v for k, v in context["source_map"].items()}
    try:
        exec(compile(source, filename, "exec"), namespace)
    except Exception as exc:
        raise failure(exc, phase="source", source=source, source_map=source_map, context=context) from exc

    class Tensor(triton.MockTensor):
        def __init__(self, binding):
            _, dtype, shape, strides, offset, alignment, _ = binding
            super().__init__(getattr(tl, D.ALL[dtype].tl_name.removeprefix("tl.")), shape)
            self.strides, self.alignment = strides, alignment

        def stride(self):
            return tuple(self.strides)

        def data_ptr(self):
            return self.alignment

    values = dict(context["scalars"], **context["consts"])
    for binding in context["bindings"]:
        if len(binding) > 2:
            values[binding[0] + "_ptr"] = Tensor(binding)
    args = [values[name] for name in context["argument_order"]]
    return preflight(namespace[context["kernel"]], args, tuple(context["grid"]), context["num_warps"],
                     source=source, source_map=source_map,
                     context=context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    print(replay(parser.parse_args().artifact))
