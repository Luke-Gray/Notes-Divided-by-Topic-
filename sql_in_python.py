##How to query in python with SQLite
import sqlite3

conn = sqlite3.connect('jobs2.db')

query = 'select Major from recent_grads order by Major desc;'

cursor = conn.cursor()
cursor.execute(query)
reverse_alphabetical = cursor.fetchall()
conn.close()

## connect sqlite to pd. important functions
DB = 'mlb.db'

def run_query(q):
    with sqlite3.connect(DB) as conn:
        return pd.read_sql(q, conn)

def run_command(c):
    with sqlite3.connect(DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        conn.isolation_level = None
        conn.execute(c)

def show_tables():
    q = '''
    SELECT
        name,
        type
    FROM sqlite_master
    WHERE type IN ("table","view");
    '''

tables = {
    'game':game,
    'park':park,
    'person':person,
    'team':team
}

with sqlite3.connect(DB) as conn:
    with name, data in tables.items():
        conn.execute('drop table if exists {};'.format(name))
        data.to_sql(name,conn,index=False)

## equivalent of head for sql query
q = """
SELECT
    game_id,
    date,
    h_name,
    number_of_game
FROM game_log
LIMIT 5;
"""


c1 = """
CREATE TABLE IF NOT EXISTS park (
    park_id TEXT PRIMARY KEY,
    name TEXT,
    nickname TEXT,
    city TEXT,
    state TEXT,
    notes TEXT
);
"""

c2 = """
INSERT OR IGNORE INTO park
SELECT
    park_id,
    name,
    aka,
    city,
    state,
    notes
FROM park_codes;
"""

## drop multiple tables
tables = [
    "game_log",
    "park_codes",
    "team_codes",
    "person_codes"
]

for t in tables:
    c = '''
    DROP TABLE {}
    '''.format(t)
    
    run_command(c)

show_tables()

## In cases where there will be multiple users or performance is important, 
## PostgreSQL is the most commonly used database engine.

## connecting to Postgre using python client
import psycopg2
conn = psycopg2.connect("dbname=postgres user=postgres")
cur = conn.cursor()

## close clinet connection when done executing queries
conn.close()

##autocommitting so that updates happen immediately
conn = psycopg2.connect("dbname=dq user=dq")
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE TABLE notes(id integer PRIMARY KEY, body text, title text)")

##example inserting values, printing fetchall() and closing
conn = psy.connect('dbname=dq user=dq')
cur = conn.cursor()
cur.execute('INSERT into notes values(1,"Do more missions on Dataquest.", "Dataquest reminder")')
rows = cur.fetchall()
print(rows)
conn.commit()
conn.close()

##creating a database and an owenr attached
conn = psy.connect('dbname=dq user=dq')
conn.autocommit=True
cur = conn.cursor()
cur.execute('CREATE DATABASE income OWNER dq;')
conn.close()

## creating a user/superuser to access db and ability to create db
CREATE ROLE sec WITH LOGIN CREATEDB PASSWORD 'test';
CREATE ROLE aig WITH LOGIN PASSWORD 'test' SUPERUSER;

## granting and revoking different privileges to user
GRANT ALL PRIVILEGES ON deposits TO sec;
REVOKE ALL PRIVILEGES ON deposits FROM sec;

##return a list of tuples by using schema for-loop
schema = conn.execute("pragma table_info(facts);").fetchall()
for s in schema:
    print(s)

##explain query plan returning a tuple
query_plan_one = conn.execute('EXPLAIN QUERY PLAN SELECT * FROM facts WHERE area = 40000;').fetchall()
query_plan_two = conn.execute('EXPLAIN QUERY PLAN SELECT area FROM facts where area>40000;').fetchall()
query_plan_three = conn.execute('EXPLAIN QUERY PLAN SELECT * FROM facts WHERE name = "Czech Republic";').fetchall()

RETURNS: [(0, 0, 0, 'SCAN TABLE facts')] ##because in each case the query has to scan the entire 'facts' table.

#If we were instead interested in a row with a specific id value, it doesnt have to go thru all rows
# since 'id' is the primary key the table is organized by and can perform a binary search

query_plan_four = conn.execute('explain query plan select * from facts where id = 20;').fetchall()
[(0, 0, 0, 'SEARCH TABLE facts USING INTEGER PRIMARY KEY (rowid=?)')]

## creating index to reduce time complexity although compromises space index
conn.execute('CREATE INDEX IF NOT EXISTS pop_idx ON facts(population);')
query_plan_seven = conn.execute('explain query plan select * from facts where population>10000;').fetchall()

##multi-index. first column is the primary key. Only jumps to rows where both conditions are met.
conn.execute('CREATE INDEX IF NOT EXISTS pop_pop_growth_idx ON facts(population, population_growth;')
RETURN [(0, 0, 0, 'SEARCH TABLE facts USING INDEX pop_pop_growth_idx (population>?)')]

##COVERING INDEX, when returning only the column in which you've set the index on. 
#So it doesn't need to search through the table ('facts' table in this case)
conn.execute("create index if not exists pop_pop_growth_idx on facts(population, population_growth);")
query_plan_five = conn.execute('explain query plan select population from facts where population>1000000;').fetchall()

[(0, 0, 0, 'SEARCH TABLE facts USING COVERING INDEX pop_idx (population>?)')]

##API request and get. and retrive status code
response = requests.get('http://api.open-notify.org/iss-pass.json')
status_code = response.status_code

##adding params argument to get a safe response code (not 400)
parameters = {"lat": 37.78, "lon": -122.41}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
content = response.content

# We can also dump a dictionary to a string and load it.
fast_food_franchise_string = json.dumps(fast_food_franchise)
print(type(fast_food_franchise_string))
fast_food_franchise_2 = json.loads(fast_food_franchise_string)

##can get the content of a response as a python onject using json()
parameters = {"lat": 37.78, "lon": -122.41}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
# Get the response data as a Python object.  Verify that it's a dictionary.
json_data = response.json()
print(type(json_data))
print(json_data)
json_datadict (<class 'dict'>)
{'message': 'success',
 'request': {'altitude': 100,
  'datetime': 1441417753,
  'latitude': 37.78,
  'longitude': -122.41,
  'passes': 5},
 'response': [{'duration': 369, 'risetime': 1441456672},
  {'duration': 626, 'risetime': 1441462284},
  {'duration': 581, 'risetime': 1441468104},
  {'duration': 482, 'risetime': 1441474000},
  {'duration': 509, 'risetime': 1441479853}]}

first_pass_duration = json_data["response"][0]["duration"]
RETURNS 369

#couting number of people in space right now
# Call the API here.
response = requests.get("http://api.open-notify.org/astros.json")

json_data = response.json()
in_space_count = json_data['number']

#weighted sum function
def weighted_mean(distribution, weights):
    weighted_sum = []
    for mean, weight in zip(distribution, weights):
        weighted_sum.append(mean * weight)
    
    return sum(weighted_sum) / sum(weights)

#mean squared distance
def variance(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
        
    return sum(distances) / len(distances)

#standard deviation
from math import sqrt
C = [1,1,1,1,1,1,1,1,1,21]

def standard_deviation(a):
    mean = sum(a)/len(a)
    diff = []
    for i in a:
        diff.append(abs(i-mean)**2)
    return sqrt(sum(diff)/len(diff))

#REASONS FOR NORMALIZATION
#1) minimize duplicate data
#2) to minimize or avoid data modification issues,
#3)	 to simplify queries.
