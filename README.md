# MA-PINN
Martingale Augmented-Physics Informed Neural Network

The MA-PINN folder contains two main subfolders:

1. B-S PIRL uses the PIRL (Physic Informed Residual Learning) method to numerically solve the B-S model from 1D to nD, as well as a comparison of layer and hidden modifications.

2. MACode presents the authors' proposed improved method MA-PINN (or MA-PIRL) method, comparing it with PINN (or PIRL).
   The Ablation Study subfolder contains experiments with the addition of hidden, layer, and epochs to PINN.
   More PDEs contains numerical solutions to several other PDEs that satisfy the martingale property P, demonstrating the broad applicability of MA-PINN.
