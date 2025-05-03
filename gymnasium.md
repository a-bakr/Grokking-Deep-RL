| Aspect                     | Gym API                                              | Gymnasium API                                           |
|-----------------------------|------------------------------------------------------|---------------------------------------------------------|
| **Reset**                   | `obs = env.reset()`                                  | `obs, info = env.reset()`                               |
| **Step**                    | `obs, reward, done, info = env.step(action)`         | `obs, reward, terminated, truncated, info = env.step(action)` |
| **Done Flag**               | Single `done` flag for end of episode                | Two flags: `terminated` (natural end) and `truncated` (time limit end) |
| **Render Modes**            | Often `env.render(mode="human")`                     | Explicit at environment creation: `gymnasium.make(env_id, render_mode="human")` |
| **Seeding**                 | `env.seed(seed)`                                     | `env.reset(seed=seed)` (during reset, not separate)      |
| **Observation and Action Spaces** | `env.observation_space`, `env.action_space`    | Same (no change)                                        |
| **Info Dictionary**         | Basic `info` returned on step                        | More detailed `info` dictionaries (e.g., why terminated) |
| **Versioning**              | Less strict                                          | Strict environment versioning, e.g., `"CartPole-v1"`     |
