import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
import logging
from openpi.training import config as configs
run = Path(os.environ["B601_RUN_DIR"])
spec = importlib.util.spec_from_file_location("b601_train", run / "train_instrumented.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def main():
    logging.basicConfig(level=logging.INFO)
    config = dataclasses.replace(configs.get_config("pi05_b601_lora"), exp_name="formal_20261120_01", resume=True, wandb_enabled=False)
    (run / "resolved_config.txt").write_text(repr(config))
    def status(state, **extra):
        payload = {"state":state, "time_unix":time.time(), "pid":os.getpid(), **extra}
        temp=run / "status.tmp"
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        temp.replace(run / "status.json")
    status("starting", target_steps=config.num_train_steps)
    try:
        module.main(config)
    except BaseException as exc:
        status("failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    else:
        status("complete", target_steps=config.num_train_steps)

if __name__ == "__main__":
    main()
