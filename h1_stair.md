# 修改urdf避免碰撞

![image](https://github.com/user-attachments/assets/9679dee0-ddd2-405c-a46d-0a64c96f3ab4)

* 显存还是上升


# hf_plane稳定训练
* 指令
```bash
python  legged_gym/legged_gym/scripts/train.py  --task=h1_45_hf  --headless  --max_iterations 2000 --sim_dev  cuda:2 --rl_device  cuda:3
```
* hf plane的配置代码

https://github.com/explore-code-world7/versa_legged_gym/blob/main/legged_gym/legged_gym/utils/terrain/plane_terrain.py

# stair训练问题
1. 执行如下命令训练h1_45_stair
```bash
python legged_gym/legged_gym/scripts/train.py --task=h1_45_stair --headless --max_iterations 1000 --sim_dev cuda:0 --rl_device cuda:1 \\
  --load_run=Jul06_11-15-26_ --checkpoint -1
```

* 开始时显存占用

![图片](https://github.com/user-attachments/assets/26cf842e-cf15-4805-88ff-3a01ea2c408f)


* 随着训练过程显存消耗上升
```bash
 Device 0 [NVIDIA RTX A6000] PCIe GEN 4@16x RX: 4.150 MiB/s TX: 50.00 KiB/s Device 1 [NVIDIA RTX A6000] PCIe GEN 4@16x RX: 300.0 KiB/s TX: 500.0 KiB/s Device 2 [NVIDIA RTX A6000] PCIe GEN 4@16x RX: 16.50 MiB/s TX: 255.2 MiB/s
 GPU 1875MHz MEM 7600MHz TEMP  67°C  FAN  39%   POW 177 / 300 W             GPU 1800MHz MEM 7600MHz TEMP  44°C  FAN  30%   POW  72 / 300 W             GPU 1800MHz MEM 7600MHz TEMP  62°C  FAN  32%   POW 133 / 300 W
 GPU[||||||||||||||               44%] MEM[||||||||      12.287Gi/47.988Gi] GPU[                              0%] MEM[||||           5.893Gi/47.988Gi] GPU[||||||||||||                 39%] MEM[||||           6.362Gi/47.988Gi]

 Device 3 [NVIDIA RTX A6000] PCIe GEN 4@16x RX: 50.00 KiB/s TX: 100.0 KiB/s Device 4 [NVIDIA RTX A6000] PCIe GEN 1@16x RX: 50.00 KiB/s TX: 200.0 KiB/s Device 5 [NVIDIA RTX A6000] PCIe GEN 1@16x RX: 300.0 KiB/s TX: 50.00 KiB/s
 GPU 1905MHz MEM 7600MHz TEMP  47°C  FAN  30%   POW  92 / 300 W             GPU 210MHz  MEM 405MHz  TEMP  30°C  FAN  30%   POW   8 / 300 W             GPU 0MHz    MEM 405MHz  TEMP  27°C  FAN  30%   POW   7 / 300 W
 GPU[                              0%] MEM[|||||          8.153Gi/47.988Gi] GPU[                              0%] MEM[               0.462Gi/47.988Gi] GPU[                              0%] MEM[               0.462Gi/47.988Gi]

 Device 6 [NVIDIA RTX A6000] PCIe GEN 1@16x RX: 2.100 MiB/s TX: 50.00 KiB/s Device 7 [NVIDIA RTX A6000] PCIe GEN 1@16x RX: 50.00 KiB/s TX: 50.00 KiB/s
 GPU 0MHz    MEM 405MHz  TEMP  26°C  FAN  30%   POW  11 / 300 W             GPU 0MHz    MEM 405MHz  TEMP  25°C  FAN  30%   POW   5 / 300 W
 GPU[                              0%] MEM[               0.462Gi/47.988Gi] GPU[                              0%] MEM[               0.462Gi/47.988Gi]
   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
100│GPU0 %                  │100│GPU1 %                  │100│GPU2 %                  │100│GPU3 %                  │100│GPU4 %                  │100│GPU5 %                  │100│GPU6 %                  │100│GPU7 %                  │
   │GPU0 mem%               │   │GPU1 mem%               │   │GPU2 mem%               │   │GPU3 mem%               │   │GPU4 mem%               │   │GPU5 mem%               │   │GPU6 mem%               │   │GPU7 mem%               │
   │                        │   │                        │   │                        │   │                        │   │                        │   │                        │   │                        │   │                        │
 75│                        │ 75│                        │ 75│                        │ 75│                        │ 75│                        │ 75│                        │ 75│                        │ 75│                        │
   │            ┌─┐     ┌─┐ │   │                        │   │                        │   │                        │   │                        │   │                        │   │                        │   │                        │
   │            │ │     │ │ │   │                        │   │                        │   │        ┌─┐             │   │                        │   │                        │   │                        │   │                        │
 50│────────────┘ └─────┘ └─│ 50│                        │ 50│                        │ 50│        │ │             │ 50│                        │ 50│                        │ 50│                        │ 50│                        │
   │                        │   │                        │   │────┐ ┌─┐ ┌───┐ ┌───────│   │        │ │             │   │                        │   │                        │   │                        │   │                        │
   │────────────────────────│   │                        │   │    └─┘ │ │   └─┘       │   │        │ │         ┌─┐ │   │                        │   │                        │   │                        │   │                        │
 25│                        │ 25│              ┌─┐       │ 25│        └─┘             │ 25│────────┼─┼─────────┼─┼─│ 25│                        │ 25│                        │ 25│                        │ 25│                        │
   │                        │   │────────────┬─┴─┼───────│   │────────────────────────│   │        │ │         │ │ │   │                        │   │                        │   │                        │   │                        │
  0│                        │  0│────────────┘   └───────│  0│                        │  0│────────┘ └─────────┘ └─│  0│────────────────────────│  0│────────────────────────│  0│────────────────────────│  0│────────────────────────│
   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘   └12s──9s────6s────3s───0s┘
    PID    USER DEV     TYPE  GPU        GPU MEM    CPU  HOST MEM Command                                                                                                                                                                      
2404023 chenlei   0  Compute  44%  12102MiB  25%   224%   6829MiB python legged_gym/legged_gym/scripts/train.py --task=h1_45_stair --headless --max_iterations 1000 --sim_dev cuda:0 --rl_device cuda:1 --load_run=Jul06_11-15-26_ --checkpoint
2399091 chenlei   3  Compute   0%   7868MiB  16%     0%   7394MiB python legged_gym/legged_gym/scripts/train.py --task=h1_45_hf --headless --max_iterations 2000 --sim_dev cuda:2 --rl_device cuda:3
2399091 chenlei   2  Compute  34%   6034MiB  12%   230%   7394MiB python legged_gym/legged_gym/scripts/train.py --task=h1_45_hf --headless --max_iterations 2000 --sim_dev cuda:2 --rl_device cuda:3
2404023 chenlei   1  Compute   0%   5554MiB  11%  2148%   6829MiB python legged_gym/legged_gym/scripts/train.py --task=h1_45_stair --headless --max_iterations 1000 --sim_dev cuda:0 --rl_device cuda:1 --load_run=Jul06_11-15-26_ --checkpoint
```

* 随着CUDA:0显存消耗不断上升，最终会出现memory access error!
```bash
/buildAgent/work/99bede84aa0a52c2/source/physx/src/NpScene.cpp (3509) : internal error : PhysX Internal CUDA error. Simulation can not continue!

/buildAgent/work/99bede84aa0a52c2/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp (11310) : internal error : GPU compressContactStage1 fail to launch kernel stage 1!!


/buildAgent/work/99bede84aa0a52c2/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp (11347) : internal error : GPU compressContactStage2 fail to launch kernel stage 1!!


[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 4202
[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 4210
[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 3480
[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 3535
[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 6137
[Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysXCuda.cu: 991
Traceback (most recent call last):
  File "legged_gym/legged_gym/scripts/train.py", line 51, in <module>
    train(args)
  File "legged_gym/legged_gym/scripts/train.py", line 47, in train
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
  File "/home/chenlei/Project/versa_legged_gym/rsl_rl/rsl_rl/runners/on_policy_runner.py", line 116, in learn
    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
  File "/home/chenlei/Project/versa_legged_gym/rsl_rl/rsl_rl/runners/on_policy_runner.py", line 154, in rollout_step
    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
  File "/home/chenlei/Project/versa_legged_gym/legged_gym/legged_gym/envs/base/legged_robot.py", line 136, in step
    self.post_decimation_step(dec_i)
  File "/home/chenlei/Project/versa_legged_gym/legged_gym/legged_gym/envs/base/legged_robot.py", line 204, in post_decimation_step
    self.substep_torques[:, dec_i, :] = self.torques
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```



