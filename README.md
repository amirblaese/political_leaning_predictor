# 🔮 political_leaning_predictor

A simple machine learning approach to predicting reddit users' political leanings.

## About

A short script using a home-trained model to predict a reddit user's political tendencies.


## Method

Step 1) Data (in the form of a large number of comments) is collected using the [reddit digger](https://github.com/amirblaese/reddit_digger)  from two subreddits broadly representing the "left" and "right" political wings. These subreddits are [/r/socialism](http://reddit.com/r/socialism) and [/r/conservative](http://reddit.com/r/conservative). 

Step 2) Once the .csv has been saved, it is loaded into the script and must be trimmed before being used to train the model. This includes:
* Removing downvoted comments (not in agreement with subreddit values, aka the hivemind principle). This threshold is set to -1.
* Removing -removed- comments.
* Tag each <.csv> file with the subreddit, i.e. 0 = socialism, 1 = conservative.
* Finally, in order to reduce any bias between the two training categories, we force both categories to have the same length.

Step 3) Train, test split (20% testing).

Step 4) Vectorize the text and train the model (see source for details).

Step 5) Use the reddit API and its python wrapper to download a users recent comments (100 usually) and use the model to predict a leaning for each comment and finally print the mean and standard deviation of the 100 comments' predictions.

## Example


    Enter reddit username: -----
    Downloading user < ---- >'s comments...
    Complete...
    Analysing 100 comments...
    Complete...
    ===========================================
    Results:
    ===========================================
    Probability of being left-leaning:   29.09 +/- 19.0 %
    Probability of being right-leaning:   70.91 +/- 19.0 %
    ===========================================
    Your most left comment is:
    I thought people just assumed that they were talking about Scalia. Not that it was directly proven t ... score =  0.9426663517951965
    ===========================================
    Your most right comment is:
    Submission Statement: We got em. The founder of BLM participated in a fraud election in 2015!!!! You ... score =  0.99994957447052
    
## Details

The model achieves an accuracy of ~80% with both the training and testing data set which I found satisfying given the naivety of the approach. You may have noticed a large error bar on the prediction of `Probability of being left-leaning`. This is fine as our main indicator for the strength of the prediction is not just the error bar but also how split the two predicutions are, i.e. how much overlap the predictions have. In the above example, the user does not post on /r/conservatism but rather on another right-wing subreddit and thus the result is accurate. Predictions in the range of 45/55, even 60 should be interpreted as "neutral".

In reality, the model should be retrained frequently with fresh data as the model appears to predict mostly based on topic and not necessarily the interpretation of the topic and so given that topics are frequently changing based on current events, the model must keep up with the topics.

