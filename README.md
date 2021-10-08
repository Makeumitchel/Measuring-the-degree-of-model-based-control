# Measuring-the-degree-of-model-based-control

Human decision making is thought to rely on two distinct systems known as the model-based (MB), responsible for planning, and the model-free (MF), embodying habits. To retrieve the strategy used by participants, many studies have made an extensive use of a  dual  system  mixing  the  two  models.  However,  the  evidences  in  favour  of  a  MF  are either weak or controversial. Moreover, we show that the mixing weights parameters of the dual model are subject to noisy estimation. We present a new approach relying on the likelihood of a MB agent to assay the degree of MB control.

This repository hosts the code used for the joined Master thesis. Inspired from the code made by Kool et al. (available on https://osf.io/793yw/) the repository enables the investigation of the degree of Model-Based control used by Human participants on a multistage decision task. In the file /data/ one can find the data of Human participants. The file /data/model_exhaustive/wrapper.mat enables to compare three methods of measuring the degree of MB control. 'method1' relies on the mixing weights of a dual  model (Kool et al. 2018). 'method2' and 'method3' are the new methods proposed relying on the log likelihood. The file /simulation/produce_data.mat produces data from a machine agent. The chosen values of the parameters' agent define the type of agent simulated (MB simple, MBMF simple, MB exhaustive, MBMF exhaustive, MB exhaustive forget, MBMF exhaustive forget).

The fitting procedure realizing Bayesian Maximum A Posteriori estimation depends on the Matlab tool "mfit" developped by (Gershman, 2016). The mfit folder needs to be added to the main folder.
