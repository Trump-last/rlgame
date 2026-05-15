import gymnasium as gym


class CenteredLandingWrapper(gym.Wrapper):
    """
    在 LunarLander 原始 reward 基础上，增加"距离着陆台中心越近越好"的惩罚。

    每步的额外惩罚 = -penalty_coeff * |lander_x - pad_center|
    """

    def __init__(self, env, penalty_coeff=0.1):
        super().__init__(env)
        self.penalty_coeff = penalty_coeff

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 只在 LunarLander 环境中生效
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "lander") and hasattr(unwrapped, "helipad_x1"):
            lander_x = unwrapped.lander.position.x
            pad_center = (unwrapped.helipad_x1 + unwrapped.helipad_x2) / 2.0
            distance = abs(lander_x - pad_center)
            penalty = -self.penalty_coeff * distance
            reward += penalty

        return obs, reward, terminated, truncated, info
