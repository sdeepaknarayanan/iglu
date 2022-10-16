Please visit the following links for downloading the respective datasets:

- [Reddit](https://drive.google.com/drive/folders/1rq-H0XUM0BIRW9Pq5P4FMC9Xirpdx6zs?usp=sharing)
- [PPI-Large](https://drive.google.com/drive/folders/1dXd0kr39IV5jZmB2h3WBr0ugXSU1F6VE?usp=sharing)
- [Arxiv](https://drive.google.com/drive/folders/1JYK71P2dYOwxgtNxR0YaKhLhvMLve9qM?usp=sharing)
- [Flickr](https://drive.google.com/drive/folders/1apP2Qn8r6G0jQXykZHyNT6Lz2pgzcQyL?usp=sharing)
- [Cora, Citeseer, Pubmed](https://github.com/tkipf/gcn)
- [Proteins](http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip)
- [Products](http://snap.stanford.edu/ogb/data/nodeproppred/products.zip)

Once downloaded, please execute the ```makedata_<dataset_name>.py``` script with appropriate referencing to the location of the files so that you can transform it into the format required by IGLU. 

For Proteins dataset, an explicit download may not necessary. The script will itself download the dataset. For Products, kindly follow the same procedure used in ```makedata_proteins.py``` to create the dataset in the requisite format.


For Cora, Citeseer and Pubmed, download the datasets from the above linked repository. 

