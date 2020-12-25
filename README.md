# Extracting information from contract documents using spaCy
In general, contract documents includes parties, dates(execution date, expiration date, termination date and renewal date), addresses.
This script is to extract contract type, parties, dates and addresses from various types of contract documents.

# packages
 - spaCy
 - numpy
 - pandas

# additional files
geodataset has been used to train the address extraction model.
And myclass.txt contains the kinds of contract types.

# execution

python main.py -i 1.txt -c myclass.txt

