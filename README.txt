Project 2: Randomized Optimization
##################################
GT CS7641 Machine Learning, Fall 2019
Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

## Background ##
Classwork for Georgia Tech's CS7641 Machine Learning course. Project code should be published publicly for grading purposes, under the assumption that students should not plagiarize content and must do their own analysis.

These experiments compare 4 randomized search algorithms across 3 different optimization problems, as well as use 3 of the algorithms to find weights for a neural network.

## Requirements ##

* Python 3.7
* PIP
* Git

## Instructions##

* Clone this source repository from Github onto your computer using the following command:
	`git clone git@github.com:ewall/CS7641_Randomized_Optimization.git`

* From the source directory, run the following command to install the necessary Python modules:
	`pip install -r requirements.txt`
	* Note that this installs my customized version of the mlrose library.

* Here is a guide to the filenames of the Python scripts in the directory:
	* Experiments to tune parameters for a particular algorithm have names beginning with the algorithm acronym and ending with the problem name. For example, `ga_fourpeaks.py` contains the experiments on the Genetic Algorithm against the Four Peaks problem.
	* Best results across all four algorithms for each of the problems have filenames beginning with the problem name and ending with "_best". For example, `knapsack_best.py` will show the tuned results of each algorithm on the Knapsack problem.
	* `knapsack_brute.py` is a brute-force search for the Knapsack problem.
	* `fourpeaks_complexity.py` is the experiment comparing algorithmic performance on varying complexities of the Four Peaks problem.
	* `fourpeaks_iterations.py` is the experiment comparing algorithmic performance across different iterations limits.
