# Information Retrieval: Project II

## Brief Description

> This project will focus on sentiment analysis. I will be using the same dataset I took from goodreads, on graphic novels, to focus on predicting the likelihood that a review is good or bad.  To prevent what happened last time where I had 500000+ entries to use, which slowed down my program, I will reduce the set to 10000 entries.
> The code will analyze the 'review_text' to determine the sentiment, whether positive, negative, or netural, and to relay the level of accuracy to the machine learning algorithm, we will use the 'rating' field.

    0-2 -> negative 
    3   -> neutral
    4-5 -> positive

___

## JSON Stats

	Total Words: 48885758 words
	Document Population: 542338 documents
	Average Word Count: 90.1389133713662 words per document
	Max Word Count: 4999 words
	Min Word Count: 0 words

___

## JSON File Structure

> Below are the raw JSON files I will be working with.

### Graphic Novel Review Sample JSON

```json
"graphic_novel_reviews.json" (542,338 reviews)
{
    "user_id": "dc3763cdb9b2cae805882878eebb6a32",
    "book_id": "18471619",
    "review_id": "66b2ba840f9bd36d6d27f46136fe4772",
    "rating": 3,
    "review_text":
        "Sherlock Holmes and the Vampires of London
        ...
        I would have to say pass on this one. \n That artwork is good, cover is great, story is lacking
        so I am giving it 2.5 out of 5 stars.",
    "date_added": "Thu Dec 05 10:44:25 -0800 2013",
    "date_updated": "Thu Dec 05 10:45:15 -0800 2013",
    "read_at": "Tue Nov 05 00:00:00 -0800 2013",
    "started_at": "",
    "n_votes": 0,
    "n_comments": 0
}
...
```

### Graphic Novel Data Sample JSON

```json
"graphic_novels.json" (89,411 book entries)
{
    "25742454": "The Switchblade Mamma",
    "30128855": "Cruelle",
}
```

### Refinement

> I will take the JSON data from both of the above raws to generate a streamlined JSON file with the relevant fields I need.

```json
"graphic_novel_reviews.json" (10,000 reviews)
{
    "book_id": "18471619",
    "title": "Sherlock Holmes and the Vampires of London",
    "rating": 3,
    "review_text":
        "Sherlock Holmes and the Vampires of London
        ...
        I would have to say pass on this one. \n That artwork is good, cover is great, story is lacking
        so I am giving it 2.5 out of 5 stars.",
}
...
```
