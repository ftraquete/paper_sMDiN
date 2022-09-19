# paper_sMDiN

Reproducible science code for "Graph properties of Mass-Difference Networks for profiling and discrimination in untargeted metabolomics" paper published on the 22nd of July 2022 in Frontiers in Molecular Biosciences Journal.

Link to the paper: https://doi.org/10.3389/fmolb.2022.917911

## Note (Paper Correction):
#### From the MDB list used and the results obtained in the aforementioned paper, the mass used for hydroxylation (CHOH) was 29.002740 Da (CHO) instead of the correct 30.010565 Da. We apologize for this mistake shown in Table 2 in the paper. However, it is worth noting that the conclusions of the paper remain inaltered and this change only leads to minor differences in the figures of the results.

#### Here, we corrected the mass in the file 'transgroups.txt' and 'transgroups_YD.txt' to the correct 30.010565 Da. Thus, the results obtained will not be exactly equal to the figures in the paper. To build the MDiNs to use this method, this corrected mass should be used.

The files 'transgroups.txt' and 'transgroups_YD.txt' have the chemical transformations with the corrected masses to give to Cytoscape's MetaNetter and build the Mass-Difference Networks.