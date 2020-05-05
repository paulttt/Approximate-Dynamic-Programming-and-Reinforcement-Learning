# Evaluation of Value Iteration , Policy Iteration and Optimistic Policy Iteration for a maze solving algorithm

In this project we evaluated three different algorithms that will find the optimal path in a maze under different cost-to-go functions. The algorithms evaluated were:

- Value Iteration (VI)
- Policy Iteration (PI)
- Optimistic Policy Iteration (OPI)

It could be seen that all algorithms produce the same policies under the same given cost function. Nevertheless the performance of the algorithms varied in great ways. The runtimes were:

0.62917876 sec. (PI) < 0.81107712 (OPI) < 1.71692276 (VI)

We expected that the VI algorithm is the slowest from these three. Nevertheless it is surprising that OPI (m=50 steps) performed slightly worse than the PI algorithm. We could explain this that only in the first two loops the PI algorithm took more than 50 iterations to evaluate the Bellmann operator. Afterwards it took strictly less than 50 iteration for the PI algorithm to evaluate the policy. Nvertheless in more complex environments OPI could yield better performance. 

### Impact of varying gamma factor 

It could be seen that a very low gamma=0.01 yields a fals policy. This can be explained from the equation that only "short-term" rewards are considered. The higher the gamma value gets the more accurate the policy gets. With a gamma>0.95 we get the optimal policy. 
