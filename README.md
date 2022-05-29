# Input_space_expansion_Base_line
This project is an analysis to determine the gold results for the "Multi-target regression via input space expansion: treating targets as inputs.[1]" experiments. On other hand, it is a baseline for the [1] experiments. 


## What is Multioutput regression?
Multioutput regression considers the regression tasks with more than one output for the prediction. 

## Baseline
In this project, I implemented a baseline for the "Multi-target regression via input space expansion: treating targets as inputs" [1] experiments. Having a baseline for the reference experiment brings more insight into the advantages of using output as an input in the model. 
This `_multivariate` class includes three different experiments as follows;
<ol>
  <li> Normal regression </li>
  <li> Predicting output via other outputs </li>
  <li> Augmenting input with the output </li>
</ol>

### Normal regression

### Highlighted approaches
* internal cross-validation methodology.
* Treating output as an input.
* Output correlation.
* Neural network
* Random Forest.
* Bagging.
* Hyperparameter optimization

## Dataset

Besides the baseline model, the MTR datasets have been used in this project. The MTR dataset includes multi-output regression tasks with various instances and dimensions. The MTR dataset introduced by [1] also. You can find the missing value treatments based on the reference in this project.

## Reproduced SST and ERC models

The introduced SST and ERC models [1], had reproduced [here](https://github.com/samanemami/Input_space_expansion).

## References
<ol>
<li> Spyromitros-Xioufis, Eleftherios, et al. "Multi-target regression via input space expansion: treating targets as inputs." Machine Learning 104.1 (2016): 55-98. </li>
</ol>