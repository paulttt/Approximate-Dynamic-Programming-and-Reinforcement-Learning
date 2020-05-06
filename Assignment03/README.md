# Dynamic Programming algorithm to process jobs from different Queues

In this projetc we wrote a scheduling algorithm based on a finite horizon Dynamic Programming problem. 

The code can be run by navigating to the directory, where the scheduler.py file lies and executing 
`
python scheduler.py
`.
After the algorithm finished the computation a plot opens up that shows the min, max and mean cost values along each stage. 
It can be seen that the costs decrease with increasing stage.

![Alt text](/Bilder/cost_plot.png)

Policies trained for really short horizon problems perform worse than longer horizons, and since the costs is constant after some k we could stop early
