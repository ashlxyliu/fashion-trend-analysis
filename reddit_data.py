import praw
import pandas as pd
from datetime import datetime

CLIENT_ID = '9EIBpAz7H_Z4u0OnhFYpAg'
CLIENT_SECRET = '-aBgjm8C68H9VO2tY31A4R_TplPzBA'
USER_AGENT = 'Fashion Trend Analyzer'

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

def fetch_reddit_posts(subreddits, limit=100):
    all_posts = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            all_posts.append({
                'subreddit': subreddit_name,
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': post.created_utc,
                'upvote_ratio': post.upvote_ratio,
                'is_self': post.is_self
            })
    return pd.DataFrame(all_posts)

if __name__ == "__main__":
    subreddits = [
        'Fashion', 'SustainableFashion', 'MaleFashionAdvice', 'FemaleFashionAdvice',
        'Streetwear', 'ThriftStoreHauls', 'WomensStreetwear'
    ]
    
    data = fetch_reddit_posts(subreddits, limit=100)
    
    data.to_csv('reddit_posts.csv', index=False)
    print("Data saved to reddit_posts.csv!")
