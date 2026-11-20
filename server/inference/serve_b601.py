import json,logging,time
from pathlib import Path
import numpy as np
import jax
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import policy_config
from openpi.training import config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
logging.basicConfig(level=logging.INFO)
root=Path("/root/experiments/pi05_b601")
run=root/"inference"
def status(state,**kw):
    (run/"status.json").write_text(json.dumps({"state":state,"time_unix":time.time(),**kw},indent=2)+"\n")
status("loading")
cfg=config.get_config("pi05_b601_lora")
policy=policy_config.create_trained_policy(cfg,root/"checkpoints/pi05_b601_lora/formal_20261120_01/5999", sample_kwargs={"num_steps":10})
split=json.loads((root/"dataset/episode_split.json").read_text())
episode=split["validation_episode_indices"][0]
dataset=LeRobotDataset("local/b601_demonstrations",episodes=[episode],video_backend="pyav")
sample=dataset[0]
images={key:np.rint(sample["observation.images."+key].numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8) for key in ["camera1","camera2"]}
obs={"images":images,"state":sample["observation.state"].numpy().astype(np.float32),"prompt":sample["task"]}
np.savez_compressed(run/"offline_observation.npz",camera1=images["camera1"],camera2=images["camera2"],state=obs["state"],prompt=np.array(obs["prompt"]))
class CheckedPolicy:
    def infer(self, observation):
        state=np.asarray(observation["state"],dtype=np.float32)
        if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError("Expected finite 7D state in radians")
        if not isinstance(observation.get("prompt"),str) or not observation["prompt"].strip():raise ValueError("Task prompt required")
        for key,shape in [("camera1",(480,640,3)),("camera2",(360,640,3))]:
            image=np.asarray(observation["images"][key])
            if image.shape!=shape or image.dtype!=np.uint8:raise ValueError(f"Expected {key} RGB uint8 {shape}")
        result=policy.infer(observation)
        actions=np.asarray(result["actions"],dtype=np.float32)
        if actions.shape!=(16,7) or not np.isfinite(actions).all():raise ValueError("Invalid model actions")
        return {"actions":actions,"action_units":"radians","model_id":"pi05_b601_lora_6000","policy_timing":result.get("policy_timing",{})}
checked=CheckedPolicy();times=[]
for i in range(4):
    start=time.monotonic();out=checked.infer(obs);elapsed=time.monotonic()-start;times.append(elapsed)
    logging.info("OFFLINE_INFERENCE round=%s seconds=%.4f shape=%s",i,elapsed,out["actions"].shape)
report={"episode":episode,"source_split":"validation","seconds":times,"actions_shape":list(out["actions"].shape),"actions_finite":bool(np.isfinite(out["actions"]).all()),"units":"radians","devices":[str(x) for x in jax.devices()],"robot_connected":False}
(run/"offline-report.json").write_text(json.dumps(report,indent=2)+"\n")
metadata={"model_id":"pi05_b601_lora_6000","action_horizon":16,"action_dim":7,"state_units":"radians","action_units":"radians","robot_connected":False}
status("ready",**metadata)
WebsocketPolicyServer(checked,host="127.0.0.1",port=8015,metadata=metadata).serve_forever()
