# Data folder

The dataset isn't committed to this repo (it's not ours to redistribute).
Download it before running the notebook:

1. Go to https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
   (free Kaggle account required)
2. Click **Download** — you'll get a file called `Tweets.csv`
3. Rename it to `airline_tweets.csv` and place it in this folder

Expected columns include: `tweet_id`, `airline_sentiment`, `negativereason`,
`airline`, `text`, `tweet_created`. The notebook uses `text` as input and
`negativereason` as the label (filtered to non-null, excluding `"Can't Tell"`).
