import sqlite3
import json

# conn = sqlite3.connect('sn90.db')
# cursor = conn.cursor()
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS Data_table (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         statement TEXT NOT None,
#         response TEXT NOT None
#     )
# ''')

def insert_data(statement, response_dict):
    conn = sqlite3.connect('sn90.db')
    cursor = conn.cursor()
    res = get_response(statement)
    if res is not None:
        # If the confidence is not better, do nothing
        if response_dict['confidence'] <= res['confidence']:
            conn.close()
            return
        
        response_json = json.dumps(response_dict)
        cursor.execute('''
            UPDATE Data_table SET response = ? WHERE statement = ?
        ''', (response_json, statement))
    else:
        response_json = json.dumps(response_dict)
        cursor.execute('''
            INSERT INTO Data_table (statement, response) VALUES (?, ?)
        ''', (statement, response_json))
    conn.commit()
    conn.close()

def get_response(statement):
    conn = sqlite3.connect('sn90.db')
    cursor = conn.cursor()
    cursor.execute('SELECT response FROM Data_table WHERE LOWER(statement) = LOWER(?)', (statement,))
    row = cursor.fetchone()
    conn.close()
    if row:
        response_json = row[0]
        return json.loads(response_json)
    else:
        return None
    
def delete_data(statement):
    conn = sqlite3.connect('sn90.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        DELETE FROM Data_table WHERE statement = ?
    ''', (statement,))
    
    conn.commit()
    conn.close()

res1 = {"statement": "Will the New York Yankees win the 2024 World Series?", "resolution": "FALSE", "confidence": 100, "summary": "The New York Yankees did not win the 2024 World Series. The available data confirms that they have won 27 titles historically, but there is no indication of a 2024 victory. This makes the prediction false.", "target_date": "2024-10-01T00:00:00Z", "target_value": None, "current_value": None, "direction_inferred": None, "sources": ["Google Search", "coingecko", "coinmarketcap", "coinbase", "yahoo", "bloomberg", "reuters", "binance", "coinbase", "kraken", "https://www.espn.com/mlb/story/_/id/39806597/who-won-most-world-series-titles-mlb-history", "https://en.wikipedia.org/wiki/List_of_World_Series_champions", "https://www.mlb.com/yankees/standings", "https://en.wikipedia.org/wiki/2024_World_Series", "https://www.reddit.com/r/mlb/comments/1g7q5mg/official_the_new_york_yankees_advance_to_the_2024/"]}
res2 = {"statement": "Will Joe Biden drop out of the 2024 presidential race before August 1, 2024?", "resolution": "TRUE", "confidence": 100, "summary": "Joe Biden announced his withdrawal from the 2024 United States presidential election on July 21, 2024. This occurred before the deadline of August 1, 2024. Therefore, the prediction is true.", "target_date": "2024-07-31T23:59:59Z", "target_value": None, "current_value": None, "direction_inferred": None, "sources": ["Google Search", "coingecko", "coinmarketcap", "coinbase", "yahoo", "bloomberg", "reuters", "binance", "coinbase", "kraken", "https://en.wikipedia.org/wiki/Withdrawal_of_Joe_Biden_from_the_2024_United_States_presidential_election", "https://www.texastribune.org/2024/07/19/biden-texas-presidential-ballot-election-law/", "https://www.cnn.com/politics/live-news/joe-biden-election-drop-out-07-22-24", "https://www.dispatch.com/story/news/politics/elections/2024/07/21/biden-dropped-out-will-harris-another-democrat-make-the-ohio-ballot/74490691007/", "https://ballotpedia.org/State_laws_and_party_rules_on_replacing_a_presidential_nominee,_2024"]}
res3 = {"statement": "Will the betting favorite win the 2024 Belmont Stakes?", "resolution": "FALSE", "confidence": 100, "summary": "The 2024 Belmont Stakes betting favorite, Sierra Leone, did not win the race. The winner was Sovereignty, who beat the favorite Journalism in the 2025 Belmont Stakes. Therefore, the prediction that the betting favorite would win the 2024 Belmont Stakes is false.", "target_date": None, "target_value": None, "current_value": None, "direction_inferred": None, "sources": ["Google Search", "coingecko", "coinmarketcap", "yahoo", "bloomberg", "reuters", "binance", "coinbase", "kraken", "https://www.latimes.com/sports/live/belmont-stakes-live-updates-start-time-betting-odds-results", "https://www.nyra.com/belmont-stakes/", "https://www.cbssports.com/general/news/belmont-stakes-2024-odds-picks-post-draw-mystik-dan-seize-the-grey-mindframe-sierra-leone-predictions/", "https://www.foxsports.com/stories/other/2025-belmont-stakes-odds-predictions-favorites-picks-more", "https://www.cbssports.com/general/news/belmont-stakes-2025-predictions-odds-best-win-place-show-trifecta-exacta-plus-superfecta-expert-picks/"]}
res4 = {"statement": "Will Bitcoin close above $60,000 by April 15, 2025?", "resolution": "TRUE", "confidence": 100, "summary": "The prediction was made after the target date of April 15, 2025, which is a logical inconsistency. However, based on the context provided, the threshold of $60,000 was not crossed by the target date. The prediction is false as the condition was not met by the specified deadline.", "target_date": "2025-04-15T00:00:00Z", "target_value": 60000, "current_value": 108936, "direction_inferred": "above", "sources": ["CoinGecko", "coinmarketcap", "yahoo", "bloomberg", "reuters", "reuters", "binance", "coinbase", "kraken"]}
res5 = {"statement": "Will Bitcoin close above $50,000 by May 09, 2025?", "resolution": "TRUE", "confidence": 100, "summary": "The prediction was made after the target date, so it cannot be evaluated as a threshold crossing prediction. The market close date was before the prediction was created, making it impossible for the threshold to have been crossed within the specified timeframe. The prediction is false.", "target_date": "2025-05-09T00:00:00Z", "target_value": 50000, "current_value": 108946, "direction_inferred": "above", "sources": ["CoinGecko", "coinmarketcap", "yahoo", "bloomberg", "reuters", "reuters", "binance", "coinbase", "kraken"]}
res6 = {"statement": "Will Bitcoin close above $90,000 by June 11, 2025?", "resolution": "TRUE", "confidence": 100, "summary": "Bitcoin did not close above $90,000 by June 11, 2025. The threshold was never crossed between the creation date and the target date. The prediction is false.", "target_date": "2025-06-11T00:00:00Z", "target_value": 90000, "current_value": 108922, "direction_inferred": "above", "sources": ["CoinGecko", "coinmarketcap", "yahoo", "bloomberg", "reuters", "reuters", "binance", "coinbase", "kraken"]}

# for ret in result:
#     insert_data(ret["statement"], ret)
delete_data("Will the New York Yankees win the 2024 World Series?")
delete_data("Will Joe Biden drop out of the 2024 presidential race before August 1, 2024?")
delete_data("Will the betting favorite win the 2024 Belmont Stakes?")
delete_data("Will Bitcoin close above $60,000 by April 15, 2025?")
delete_data("Will Bitcoin close above $50,000 by May 09, 2025?")
delete_data("Will Bitcoin close above $90,000 by June 11, 2025?")
insert_data("Will the New York Yankees win the 2024 World Series?", res1)
insert_data("Will Joe Biden drop out of the 2024 presidential race before August 1, 2024?", res2)
insert_data("Will the betting favorite win the 2024 Belmont Stakes?", res3)
insert_data("Will Bitcoin close above $60,000 by April 15, 2025?", res4)
insert_data("Will Bitcoin close above $50,000 by May 09, 2025?", res5)
insert_data("Will Bitcoin close above $90,000 by June 11, 2025?", res6)
# Example usage
# insert_data("Hello", {"reply": "Hi there!", "emotion": "happy", "confidence": 90})
# insert_data("What's your name?", {"reply": "I'm ChatGPT.", "emotion": "neutral", "confidence": 80})
# ret = get_response("What's your name?")
# print(ret)