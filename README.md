# Reinforcement Learning
Implementation and experiments with reinforcement learning algorithms and testing them with OpenAI's gym [[1]](https://arxiv.org/pdf/1606.01540.pdf).

## Success Matching
> "When learning from a set of their own trials in iterated decision problems, humans attempt to match not the best taken action but the reward-weighted frequency of their actions and outcomes"
Arrow, 1958

Episode-based successs matching is a probabilistic policy search method. A parameter distribution is learned by iteratively fitting the reward weighted parameter distribution. This happens in multiple steps:
1. Explore: Sample parameters from the parameter distibution
2. Evaluation: Evaluate the resulting policy by collecting rewards from the environment
3. Update: Fit the reward weighted parameter distribution with with weighted maximum likelihood

This algorithm is known as policy learning by weighting exploration with the returns (PoWER, [[2]](./papers/NIPS-2008-policy-search-for-motor-primitives-in-robotics-Paper.pdf)).

### Cart Pole
For the [cart pole environment](https://www.gymlibrary.ml/pages/environments/classic_control/cart_pole) PoWER works just fine with a time-independent linear policy model and no featurization of the state.
The first clip shows the policy bevore training and then the policy with the mean of the paramter distribution for each epoch. The training runs for 10 epochs with 5 parameter samples and rollouts per epoch. 




https://user-images.githubusercontent.com/17069602/159137319-7327e3a0-26cd-454f-8813-cceca16320d5.mp4



### Pendulum
For the [pendulum environment](https://www.gymlibrary.ml/pages/environments/classic_control/pendulum) PoWER works with a time-independent policy with a featurization of the state. The policy is linear in its parameters. Hurdles of the pendulum environment are that the state is given as the positions in task space and not in joint space with a non-linear dynamics model and the heterogeneity in the start position. For the first hurdle the joint angle was calculated and for the second hurdle the start position was seeded for all rollouts in one epoch. The last clip shows the pendulum starting in different positions and the learned policy is able to solve them but only uses one side for the upswing. This problem could be solved with hierarchical relative entropy policy search (HiREPS, [[3]](https://is.mpg.de/fileadmin/user_upload/files/publications/2012/DanielAISTATS2012.pdf)) which uses a gating-policy for an option dependent on the state and an option-policy for the policy parameters dependent on the option. In this manner HiREPS can learn versatile solutions.




https://user-images.githubusercontent.com/17069602/159137207-2f7c72ec-fca7-4b97-8002-6b5360147591.mp4




## References
[[1]](https://arxiv.org/pdf/1606.01540.pdf) G. Brockman and V. Cheung and L. Pettersson and J. Schneider and J. Schulman and J. Tang and W. Zaremba, "OpenAI Gym", 2016

[[2]](./papers/NIPS-2008-policy-search-for-motor-primitives-in-robotics-Paper.pdf) J. Kober and J. Peters, “Policy search for motor primitives in robotics”,
in Advances in Neural Information Processing Systems (NIPS), 2008.

[[3]](https://is.mpg.de/fileadmin/user_upload/files/publications/2012/DanielAISTATS2012.pdf) C. Daniel, G. Neumann, and J. Peters. "Hierarchical relative entropy policy search", in Artificial Intelligence and Statistics, pages 273–281, 2012.
