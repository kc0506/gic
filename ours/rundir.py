# coding=utf-8
"""Auto-placed run directory for the fit_traj_* / fit_image_* entrypoints.

gic-side analogue of reuse_mpm's run_io.RunDir (can't reuse that one -- different
repo, different output tree). Runs auto-land at  output/<task>/<NN>[_<label>]/
where <task> is the entrypoint module name and NN auto-increments, so `ls` shows
run order without timestamps. The tyro-parsed config dataclass is auto-saved to
config.json (entrypoints no longer hand-write config_used.json). Pass out= to
override placement (scratch escape hatch).

Console capture is left to the launcher (`python fit_x.py ... > run.log 2>&1`),
matching the existing gic convention; no FD-tee here.
"""
import json
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass


@dataclass
class RunDir:
    root: str

    @classmethod
    def create(cls, module: str, label: str = "", out: "str | None" = None,
               config=None, out_root: str = "output") -> "RunDir":
        """module: the entrypoint's __name__; label: optional human suffix.

        If config (a dataclass) is given its resolved values are auto-saved to
        config.json right here. out, if given, is used verbatim.
        """
        if module == "__main__":  # `python fit_x.py` -> __name__ is "__main__"
            module = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        if out:
            root = out
        else:
            task = module.rsplit(".", 1)[-1]               # e.g. "fit_traj_Escalar"
            base = os.path.join(out_root, task)
            os.makedirs(base, exist_ok=True)
            nns = [int(d[:2]) for d in os.listdir(base)
                   if len(d) >= 2 and d[:2].isdigit()]
            nn = max(nns, default=-1) + 1
            root = os.path.join(base, f"{nn:02d}_{label}" if label else f"{nn:02d}")
        os.makedirs(root, exist_ok=True)
        rd = cls(root)
        if config is not None:
            rd.save_config(config)
        return rd

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def save_config(self, config, **extra) -> None:
        """Auto-save the config dataclass (+ derived extras) to config.json."""
        d = asdict(config) if is_dataclass(config) else dict(config)
        with open(self.path("config.json"), "w") as f:
            json.dump({**d, **extra}, f, indent=2, default=str)
