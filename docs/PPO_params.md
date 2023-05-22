
| ID | Name                                                     | Default Vaule             |
|:---|:---------------------------------------------------------|:--------------------------|
| 1  | num envs iteration size num_epochs batch size batch mode | 256                       |
| 2  | iteration_size                                           | 2048                      |
| 3  | num_epochs                                               | 10                        |
| 4  | batch_size                                               | 64                        |
| 5  | batch_mode                                               | Shuffle transitions       |
| 6  | advantage_estimator                                      | GAE                       |
| 7  | GAE Y                                                    | 0.95                      |
| 8  | Value function loss                                      | MSE                       |
| 9  | PPO-style value clipping ε                               | 0.2                       |
| 10 | Policy loss                                              | PPO                       |
| 11 | PPO ε                                                    | 0.2                       |
| 12 | Discount factor γ                                        | 0.99                      |
| 13 | Frame skip                                               | 1                         |
| 14 | Handle abandoned?                                        | False                     |
| 15 | Optimizer                                                | Adam                      |
| 16 | Adam learning rate                                       | 3е-4                      |
| 17 | Adam momentum                                            | 0.9                       |
| 18 | Adam ε                                                   | 1e-7                      |
| 19 | Learning rate decay                                      | 0.0                       |
| 20 | Regularization type                                      | None                      |
| 21 | Shared MLPs?                                             | Shared                    |
| 22 | Policy MLP width                                         | 64                        |
| 23 | Value MLP width                                          | 64                        |
| 24 | Policy MLP depth                                         | 2                         |
| 25 | Value MLP depth                                          | 2                         |
| 26 | Activation                                               | tanh                      |
| 27 | Initializer                                              | Orthogonal with gain 1.41 |
| 28 | Last policy layer scaling                                | 0.01                      |
| 29 | Last value layer scaling                                 | 1.0                       |
| 30 | Global standard deviation?                               | True                      |
| 31 | Standard deviation transformation Tρ                     | safe exp                  |
| 32 | Initial standard deviation iρ                            | 1.0                       |
| 33 | Action transformation Tu                                 | clip                      |
| 34 | Minimum standard deviation ερ                            | le-3                      |
| 35 | Input normalization                                      | Average                   |
| 36 | Input clipping                                           | 10.0                      |
| 37 | Value function normalization                             | Average                   |
| 38 | Per minibatch advantage normalization                    | False                     |
| 39 | Gradient clipping                                        | 0.5                       |