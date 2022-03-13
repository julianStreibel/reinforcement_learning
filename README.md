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
For the [cart pole environment](https://www.gymlibrary.ml/pages/environments/classic_control/cart_pole) PoWER works just fine with a linear policy model and no featurization of the state.
The first clip shows the policy bevore training and then the policy with the mean of the paramter distribution for each epoch. The training runs for 10 epochs with 5 parameter samples and rollouts per epoch. 



https://user-images.githubusercontent.com/17069602/158071031-98b3625b-88a9-4d7f-b250-87210a3b79e8.mp4




## References
[[1]](https://arxiv.org/pdf/1606.01540.pdf) G. Brockman and V. Cheung and L. Pettersson and J. Schneider and J. Schulman and J. Tang and W. Zaremba, "OpenAI Gym", 2016

[[2]](./papers/NIPS-2008-policy-search-for-motor-primitives-in-robotics-Paper.pdf) J. Kober and J. Peters, “Policy search for motor primitives in robotics”,
in Advances in Neural Information Processing Systems (NIPS), 2008.
