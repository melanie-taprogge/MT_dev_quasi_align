# AI-Biome

This tool aims to find a minimal set of reliable markers for identification of given target species. Optionally, for each target species, a list of non-target closely related species may be specified.

The marker regions are searched for within single-copy core genes only.

1. Pan-genome analysis

Input genomes are annotated using Prokka (Seemann, 2014) and core genes are identified with Panaroo (Tonkin-Hill, 2020). Annotated sequences are saved to a local PostgreSQL database. Single-copy core genes are then queried from this database and saved in the multi-FASTA format.

2. Multiple sequence alignment

Single copy core genes are aligned using one of the following algorithms:

* ClustaW
* MUSCLE
* MAFFT
* probabilistic multiple alignment program PRANK (Löytynoja, 2014)

3. Identification of conserved regions

Identification of conserved regions for each species can be carried out by means of:
* Consensus sequence
* Shannon's entropy
* Quasi-alignment-based method proposed by Nagar & Hahsler (2013)

