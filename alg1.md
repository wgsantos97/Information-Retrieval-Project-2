# Algorithm 2: Naive Bayes Model

## Summary

> A few minor changes were made to the structure of the code to improve overall accuracy. Initial tests had the algorithm scoring at 20%. But the reduction of fields to binary based off the "sentiment" as well as changing the Naive Bayes model from GaussianNB to MultinomialNB increased the rating to 71%, a drastic improvement from the initial scores. The splitting of the data was also altered, such that an even number of positive and negative reviews were fed into the machine learning algorithm, reducing its bias for reading specific datasets.

## General Stats

___

### Test 1

    Data Size: 1000
    Accuracy: 62.2%
    
    Confusion Matrix
    X Length: (500, 4813)
    [ [156 98] 
      [91 155] ]

### Test 2

    Data Size: 10000
    Accuracy: 72.68%
    
    Confusion Matrix
    X Length: (5000, 22991)
    [ [1744 637] 
      [729 1890] ]

### Test 3

    Data Size: 100000
    Accuracy: 71.764%
    
    Confusion Matrix
    X Length: (50000, 103257)
    [ [18386 7577] 
      [6541 17496] ]

## Results

> After testing the code against 1000, 10000, and then 100000 pieces of data, it appears that the code maxes out at around 71%. It is unlikely that this implementation will exceed 80%. Note that the data takes over an hour to process at higher volumes (100000+), so it is highly advised not to use low-end laptops against data volumes larger than 1000.

## Notes

> For more information regarding the data stats, please look at 'results.md' located inside the 'Data' folder.
