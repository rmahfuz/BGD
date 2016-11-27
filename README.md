Fault-tolerant-distributed-optimization simulation of local estimates

Assumptions:
1) The distribution where the data is taken from is a gaussian distribution, generated from :
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html
2) loss function of a point is Euclidean distance from the center of the class, where center is the average of all points in that class
3) The local estimate at t=0 is set as 0 (update rule only tells you how to update the local estimate, but not how to initialize it)
4) gradient = half of the value (obviously that's not right), because I don't know how esle to calculate gradient of a value

Results:
Values of individual loss function for each data point for each agent (15 agents, each has 5 data points):
[0    2.405884
1    0.692434
2    0.399686
3    0.328271
4    1.717425
Name: 5, dtype: float64]
[0    1.720843
1    1.149409
2    0.685373
3    0.780379
4    1.738981
Name: 5, dtype: float64]
[0    1.560199
1    1.061070
2    0.229189
3    2.211233
4    0.441716
Name: 5, dtype: float64]
[0    1.709869
1    0.480518
2    1.843023
3    1.073688
4    0.354841
Name: 5, dtype: float64]
[0    1.709869
1    0.603661
2    2.124596
3    0.620070
4    0.634708
Name: 5, dtype: float64]
[0    1.611371
1    1.738981
2    0.959160
3    1.150207
4    0.393614
Name: 5, dtype: float64]
[0    1.613594
1    0.190092
2    0.861134
3    0.513964
4    1.717425
Name: 5, dtype: float64]
[0    1.764571
1    0.239852
2    0.365733
3    2.762231
4    1.547731
Name: 5, dtype: float64]
[0    2.167485
1    0.505669
2    1.142621
3    1.113653
4    2.701330
Name: 5, dtype: float64]
[0    1.727728
1    0.244629
2    0.461003
3    0.471444
4    1.720843
Name: 5, dtype: float64]
[0    1.613594
1    1.545044
2    0.870072
3    1.324733
4    0.329963
Name: 5, dtype: float64]
[0    1.032103
1    1.132818
2    1.727728
3    1.646402
4    0.606370
Name: 5, dtype: float64]
[0    1.861388
1    0.609218
2    0.781818
3    0.359985
4    2.028109
Name: 5, dtype: float64]
[0    1.717425
1    1.106758
2    1.543098
3    1.024161
4    2.762231
Name: 5, dtype: float64]
[0    2.124596
1    0.430776
2    0.283965
3    0.110377
4    2.762231
Name: 5, dtype: float64]
