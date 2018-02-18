# Fault-tolerant-distributed-optimization
Simulating behavior of machines in a distributed machine learning setting

Algorithm:

The Byzantine Gradient Descent algorithm(https://arxiv.org/abs/1705.05491) has been simulated here. This algorithm is similar to Standard Gradient Descent, except that in the update step, it uses the geometric median of means of gradients of different batches, instead of just the mean. This is what makes Byzantine Gradient Descent robust to byzantine faults.
The implementation of gemoetric median was found here:
https://github.com/mrwojo/geometric_median/blob/master/geometric_median/geometric_median.py

Files:

bgd.py is the file containing the actual Python code. The configuration can be manipulated in config.txt. runme.bash runs the algorithm a specific number of times (the number of times and output file can be specified as an argument). The result of running the algorithms a number of times are presented in byzantine_stats_malicious.txt and standard_stats_malicious.txt. analyze.py analyzes the results of running the algorithm multiple times.

Results so far:

Twelve machines, divided into four batches were subjected to both algorithms: Standard Gradient Descent (SGD) and Byzantine Gradient Descent (BGD). The underlying truth was (3, 4), and the value to be learned was initialized to (9, 10).
When one of the machines was byzantine, it was found that in BGD, the learned value converged to the underlying truth much more accurately than in SGD. Each algorithm was performed 30 times (using runme.bash) with 100 iterations (specified in config.txt), and the final learned value for each execution was recorded (in standard_stats_malicious.txt and byzantine_stats_malicious.txt). The average value that BGD converged to was (2.70, 3.54), while SGD converged to an average value of (1.62, -9.67). While the value learned by SGD was affected greatly by the presence of a single byzantine fault, BGD proved to be able to tolerate that fault.
The error in the learned value was 15.05 in the SGD case, while the error was only 0.76 in the BGD case.

Conclusion so far:

In the presence of a byzantine fault, the error in the learned value is reduced by 95% if Byzantine Gradient Descent is used instead of Standard Gradient Descent.

This project is supported by the Center for Science of Information.
https://www.soihub.org/education/research-teams/team-machine-learning-security/
