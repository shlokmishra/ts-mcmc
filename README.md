# ts-mcmc
This repo is a simple prototype for using tskit to store phylogenetic MCMC output.
It currently features a simple MCMC algorithm targeting the Kingman coalescent without mutation.

The project contains the following:

- mcmc.py: The specification of the MCMC loop. Operates on a tree specified in `tree.py`, with successive iterates compactly stored into a tree sequence by a recorder class specified in `recorder.py`.
- recorder.py: The class for efficiently copying MCMC iterates (stored as tree objects specified in `tree.py`) into a tree sequence.
- run.py: A short script for running one-off simulations.
- tree.py: The data structure for storing and updating an individual tree in the MCMC loop. Also contains the specifications of the Kingman coalescent target distribution, and the MCMC proposal distribution.
- verification.py: A statistical comparison of TMRCAs, total branch lengths, and youngest clade probabilities against analytical means. Set to $10^5$ MCMC iterations by default, with a runtime of around 6 minutes. Runtime is roughly linear in the number of iterations.
