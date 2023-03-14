# scalable-gaussian-processes

## Project next steps

### Ablations — UCI datasets 

* Choose hyperparams with exact MLL
* Estimator variance 
	* Compare sampling objectives in terms of variance
	* Compare initialisations
	* Investigate effect of rff for regulariser estimation
	* Instigate effect of RFF for prior function sampling 
	* Check changes in estimator variance agree with theory 
* Sampling accuracy 
	* Exact vs Inducing point vs RFF vs CG vs our method
	* Look at Gibbs ringing in RFF
	* Look at difference in time between CG solves and our solves. Look at error vector 

### Large scale benchmarks

* Use “million datapoint GP” datasets — try to use their hyperparams
* Sample with CG (marginal or pathwise), VI (marginal or pathwise) and our method

### Application demonstration
Still need to decide between

* Glacier
* Medical
* Molecular discovery in a latent space 

