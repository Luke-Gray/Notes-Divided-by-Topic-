###return current directory
import os
os.getcwd()

## print all file names in your current directory
import os
os.listdir("somedirectory")

##Return true if file exists, else False
import os.path
os.path.isfile(fname) 

##To be sure to its a file
from pathlib import Path

my_file = Path("/path/to/file")
if my_file.is_file():
    # file exists

#to check if a directory exists
if my_file.is_dir():
    # directory exists

#Additional option
import os
import os.path

PATH='./file.txt'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print "File exists and is readable"
else:
    print "Either the file is missing or not readable"


##calling an external command from Python. subprocess provides more powerful facilities for
# spawning new processes and retrieving their results
import subprocess
subprocess.run(["ls", "-l"])

#Simple mode with array of numbers
def SimpleMode(a):
    m = [1, -1]
    for i in a:
        if a.count(i) > m[0]:
            m[0] = a.count(i)
            m[1] = i
    return m[1]

#Making a copy of a list in order to remove items from it, for if you don't copy you will interrupt looping thru it.
ex = np.array([5,10,15])
b=[]

for i in range(len(ex[1:])):
    while ex[i]<ex[i+1]:
        ex[i] = ex[i]+1
        b.append(ex[i])

for i in b:
    for i in b:
        if i in ex:
            b.remove(i)
        
print(len(b))

##Using find to place commas
def FormattedDivision(num1,num2): 
    s = str(round(float(num1) / num2*10000)/10000)
    if s.find(".") > 0:
        s += "0000"
        s = s[:s.find(".") +5]
    else:
        s += '.0000' 
    if s.find(".") > 3:
        s = s[:s.find(".") -3] + "," + s[s.find(".") -3:]
    while s.find(",") > 3 : 
        s = s[:s.find(",") -3] + "," + s[s.find(",") -3:]        
    return s

##Multiplying a matrix by its inverse to return an identity matrix
matrix_a = np.asarray([
[1.5, 3],
[1, 4]
])
def matrix_inverse_two(mat):
    det = (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])
    if det == 0:
        raise ValueError("The matrix isn't invertible")
    right_mat = np.asarray([
        [mat[1,1], -mat[0,1]],
        [-mat[1,0], mat[0,0]]
    ])
    inv_mat = np.dot(1/det, right_mat)
    return inv_mat

inverse_a = matrix_inverse_two(matrix_a)

i_2 = np.dot(inverse_a, matrix_a)


##one-liner to invert a matrix
 return n.linalg.inv(matrix)

 #converting an equation from standard form to slope-intercept form
 8x + 4y = 5
 2x + y = 2

 y1 = -2*x + (5/4)
 y2 = -2*x + (5/2)	

 ##OLS format using linalg and np.notations
first_term = np.linalg.inv(
        np.dot(
            X.T, 
            X
        )
    )
second_term = np.dot(
        X.T,
        y
    )
ols_estimation = np.dot(first_term, second_term)

#Syntax : maketrans(str1, str2, str3)
Parameters :
str1 : Specifies the list of characters that need to be replaced.
str2 : Specifies the list of characters with which the characters need to be replaced.
str3 : Specifies the list of characters that needs to be deleted.

#Returns : Returns the translation table which specifies the conversions that can be used by translate()
string = 'weeksyourweeks'
st1 = 'wy'
st2 = 'gf'
st3 = 'u'
answer = string.maketrans(st1,st2,st3)
print(answer)
translation = string.translate(answer)
print(translation)

#MostFreeTime Coding Challenge
def turn_minutes(time):
    result = time[:-2].split(":") 
    result = [int(result[0]), int(result[1])] if time[-2:] == "AM" else [int(result[0])+12, int(result[1])]
    result[0] = result[0] if (result[0]!=12 and result[0]!=24) else result[0]-12
    return result[0] * 60 + result[1]
def MostFreeTimes(strArr): 
    times_list, most_free = [], 0
    for time in strArr:
        times_list.append([turn_minutes(i) for i in time.split("-")])
    times_list.sort()
    for i in range(len(times_list)-1):
  	    if times_list[i+1][0] - times_list[i][1] > most_free:
  		    most_free =  times_list[i+1][0] - times_list[i][1]
    return "%02d:%02d" %(most_free/60, most_free%60) 

##Decision tree splitting node and calculating information gain
median_age = income["age"].median()

left_split = income[income["age"] <= median_age]
right_split = income[income["age"] > median_age]

age_information_gain = income_entropy - ((left_split.shape[0] / income.shape[0]) * 
	calc_entropy(left_split["high_income"]) + ((right_split.shape[0] / income.shape[0])
	 * calc_entropy(right_split["high_income"])))

 ##grabbing value in list when looping thru
 def ArithGeoII(arr):
ac = []
gc = []
for i in arr[1:]:
    ac.append(i - (arr[arr.index(i) - 1]))
    gc.append(i / (arr[arr.index(i) - 1]))
if len(set(ac)) == 1:
    return "Arithmetic"
elif len(set(gc)) == 1:
    return "Geometric"
return -1

	## swapcase and switch order of numericals
def SwapII(s):
s = list(s.swapcase())
for i in range(len(s)):
    ##the rest of this code is to swap the numericals
    if s[i] in '1234567890':
        for x in range(i + 1, len(s)):
            if s[x] not in '1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM':
                break
            elif s[x] in '1234567890':
                l = s[i]
                s[i] = s[x]
                s[x] = l
return ''.join(s)

##Using insert function
s = '99946'
s = str(s)
s = '0' + s
a = []
print(s)
for i in range(len(s)):
    a.append(s[i])
    print(a)
    if int(s[i]) % 2 == 1 and int(s[i - 1]) % 2 == 1:
        a.insert(len(a) - 1, '-')
    elif int(s[i]) % 2 == 0 and int(s[i-1]) % 2 == 0 and int(s[i]) != 0 and int(s[i-1]) != 0:
        a.insert(len(a) - 1, '*')
print(''.join(a[1:]))

## appending list within list (as coordinates/queen position)
queen1 = np.array(["(2,1)", "(4,2)", "(6,3)", "(8,4)", "(3,5)", "(1,6)", "(7,7)", "(5,8)"])
n1 = []
for i in queen1:
    n1.append([int(i[1]), int(i[3])])

## another way to grab integers in strArr
b_string = np.array(["[5, 18]", "[1, 2, 6, 7]"])
x = [int(i) for i in b_string[0][1:-1].split(',')]

##multiple nested list grabbing the same item
s="Hello -5LOL6"

s = list(s.swapcase())
for i in range(len(s)):
    if s[i] in '1234567890':
        for x in range(i + 1, len(s)):
            print(s[x])
            if s[x] not in '1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM':
                break
            elif s[x] in '1234567890':
                l = s[i]
                s[i] = s[x]
                s[x] = l
print(''.join(s))


##grabbing all substrings that have 3 unique characters in them
s = "3aabacbebebe"
k = int(s[0])
c = []
for y in range(k,len(s) + 1):
    for x in range(y):#grabs from groups of 3 onward
        if len(set(s[x:y])) == k:
            c.append(s[x:y])
print(max(c, key=len))  


##Two different ways to grab rows and columns of multi-array
a = ["10111", "10101", "11101", "11111"]

holes, zeroes = 0, []
for row in range(len(a)):
    for column in range(len(a[row])):
        if a[row][column] == '0':
            zeroes.append([row, column])
            holes += 1
for i in zeroes:
    x, y = i[0], i[1] 

 #ORRRRR

 answer,l = 0, []

for i, row in enumerate(s):
    for j, col in enumerate(list(row)):
        if col == '0': l.append((i, j)):


 ###Indexing range using i.index and i.index + i
a = [1, 3, 6, 8, 2, 7, 1, 2, 1, 2, 6, 1, 2, 1, 2]

c = len(a[a[0]:])
f = max(a[:a[0]])
s = max(a[:a.index(f)+f])
t = max(a[s:(a.index(s)+1)+s])
f = max(a[t:(a.index(t)+1)+t])

def LetterChanges(s):
    new = []
    for i in s:
        if i in alpha:
            if alpha[alpha.index(i) + 1] in 'aeiou':
                new.append(alpha[alpha.index(i) + 1].upper())
            else:
                new.append(alpha[alpha.index(i) + 1])
        else:
            new.append(i)
    return ''.join(new)
    
## list comprehension to create multidimensional array
def SeatingStudents(a):
    l = [[i,i+1] for i in range(1,a[0]+1) if i%2!=0]
    count = 0

    for i in range(len(l)-1):
        for j in range(len(l[i])-1):
            if (l[i][j] and l[i][j+1]) not in a[1:]:
                count+=1
            elif (l[i][j] and l[i+1][j]) not in a[1:]:
                count+=1
            elif (l[i][j+1]) and (l[i+1][j+1]) not in a[1:]:
                count+=1
    return count*2


##different consideration for indexing (True if sum of all previous is less than current value)
a = [1, 3, 6, 13, 54]

for x in range(len(a))[1:]:
    if a[x] <= sum(a[:x]):
        print('false')
print('true')

##indexing through an ongoing range
def KUniqueCharacters(s):
    k = int(s[0])
    c = []
    for y in range(k,len(s) + 1):
        for x in range(y):
            if len(set(s[x:y])) == k:
                c.append(s[x:y])
    return max(c, key=len)    


##finding the substring in a[0] that contains all in a[1]
def MinWindowSubstring(a):
    N, K, m = a[0], a[1], ['', len(a[0])]
    for y in range(len(K), len(N) + 1):
        for x in range(y):
            l, c = N[x:y], list(K[:])#l is every single instance
            for i in l:
                if i in c:
                    c.remove(i)
            if len(c) == 0 and len(l) < m[1]:
                m[0], m[1] = l, len(l)
    return m[0]

#return word split by comma based on list of words in second string
import itertools
def WordSplit(s):
    a, b, c = s[0], list(s[1].split(',')), []

    for i in b:
        if i in a:
            c.append(i)
    for pair in itertools.combinations(c,2):
        if (len(pair[0])+len(pair[1])) == len(a):
            return pair[0]+','+pair[1] if a[:len(pair[0])]==pair[0] else pair[1]+','+pair[0]
    return 'not possible'

### Google-stye docstring of function and retrieving it with docstring
def count_letter(content, letter):
    '''Count the number of times `letter` appears in `content`.
    
    Args:
        content(str): string to check whether a letter appears in it.
        letter(str): single character string to check whether or not
    it appears in content.
    
    Returns:
        1(int): if letter found in content
        ValueError: if letter not in found content
    '''
    if (not isinstance(letter, str)) or len(letter) != 1:
        raise ValueError('`letter` must be a single character       string.')
    return len([char for char in content if char == letter])

formatted_docstring = inspect.getdoc(count_letter)

The only way to tell if something is mutable is to see if there is a function
 or method that will change the object without assigning it to a new variable.

If you really want a mutable variable as a default value, 
consider defaulting to None and setting the argument in the function:

def foo(var=None):
    if var is None:
        var = []
    var.append(1)
    return var

## adding a decorator from the contect module to the timer() function to make it act like a context manager
import contextlib
import time

@contextlib.contextmanager
def timer():
    """Time the execution of a context block.

    Yields:
      None
    """
    start = time.time()
    # Send control back to the context block
    yield
    end = time.time()
    print('Elapsed: {:.2f}s'.format(end - start))
    
with timer():
    print('This should take approximately 0.25 seconds')
    time.sleep(0.25)

#the ability for a function to yield control and know that it will get to finish running later
# is what makes context managers so useful.


#this context manager uses this set up connect to a database
@contextlib.contextmanager
def database(url):
    # set up database connection
    db = postgres.connect(url)

    yield db

    # tear down database connection
    db.disconnect()

url = 'http://dataquest.io/data'
with database(url) as my_db:
    course_list = my_db.execute(
      'SELECT * FROM courses'
  )
#This setup/teardown behavior allows a context manager to hide things like connecting and disconnecting
# from a database, so that a programmer using the context manager can just perform operations on the database
# without worrying about the underlying details.


#context manager to open open and read file

@contextlib.contextmanager
def open_read_only(filename):
    """Open a file in read-only mode.

    Args:
      filename (str): The location of the file to read

    Yields:
      file object
    """
    read_only_file = open(filename, mode='r')
    # Yield read_only_file so it can be assigned to my_file
    yield read_only_file
    # Close read_only_file
    read_only_file.close()

with open_read_only('my_file.txt') as my_file:
    print(my_file.read())


#copy content of one source to another source by looping through each line
def copy(src, dst):
    """Copy the contents of one file to another.

    Args:
      src (str): File name of the file to be copied.
      dst (str): Where to write the new file.
    """
    # Open both files
    with open(src) as f_src:
        with open(dst, 'w') as f_dst:
            # Read and write each line, one at a time
            for line in f_src:
                f_dst.write(line)

### using eval() to create a dict of a list of strings
strArr = ["1:[5]", "2:[5,18]", "3:[5,12]", "4:[5]", "5:[1,2,3,4]", "18:[2]", "12:[3]"]

g = eval('{' + ','.join(strArr) + '}')
e = dict()
v0 = g.keys()
g

##choosing at random, applicable to situations like choosing a random assessment
import random

mylist = ["apple", "banana", "cherry"]

print(random.choices(mylist, weights = [10, 1, 1], k = 14))

##z-score -- easier to intepret distance from mean. can be positive negative to show direction
distance = 220000 - houses['SalePrice'].mean()
st_devs_away = distance / houses['SalePrice'].std(ddof = 0)

#z-score using apply(lambda x:)
mean_area = houses['Lot Area'].mean()
stdev_area = houses['Lot Area'].std(ddof = 0)
houses['z_area'] = houses['Lot Area'].apply(
    lambda x: ((x - mean_area) / stdev_area)
    )

#When we convert a distribution to z-scores, we'd say in statistical jargon
# that we standardized the distribution.

##coin toss function and current probabilities
seed(1)

def coin_toss():
    if randint(0,2) == 1:
        return 'HEAD'
    else:
        return 'TAIL'
    
probabilities = []
heads = 0

for n in range(1, 10001):
    outcome = coin_toss()
    if outcome == 'HEAD':
        heads+=1
    current_probability = heads/n
    probabilities.append(current_probability)

#probability of rolling c conditions or d conditions is
# the sum of the probabilities minus the probability of both(c & d)

p_c = 3/6
p_d = 3/6
p_c_and_d = 2/6

p_c_or_d = p_c + p_d - p_c_and_d

##mutually exclusive events is just:
p_c_or_d = p_c + p_d

#P(A or B) becomes P(A ∪ B).
#P(A and B) becomes P(A ∩ B).

##1) Tuples are faster than lists. If you're defining a constant set of values and all you're ever going
## to do with it is iterate through it, use a tuple instead of a list.

##2) It makes your code safer if you “write-protect” data that does not need to be changed.
## Using a tuple instead of a list is like having an implied assert statement that this data is constant,
# and that special thought (and a specific function) is required to override that.

##3)Some tuples can be used as dictionary keys (specifically, tuples that contain immutable values like 
##strings, numbers, and other tuples). Lists can never be used as dictionary keys, because lists are not
# immutable.

##4) Can also use the 'in' operator for tuples to check if something is contained in the tuples

## Use list instead of tuples if you want to add or remove items.


##Practical way to use .pop()
fruit = [['Orange','Fruit'],['Banana','Fruit'], ['Mango', 'Fruit']] 
consume = ['Juice', 'Eat'] 
result = [] 

# Iterating item in list fruit 
for item in fruit : 
      
    # Inerating use in list consume 
    for use in consume : 
          
        item.append(use) ##appends 'Juice'
        result.append(item[:]) ##appends all of item with Juice at the end
        item.pop(-1) ##pops out juice from item, and gets ready to append 'Eat', since it exhaust consume 
print(result) 		## before looping to the next item in fruit

[['Orange', 'Fruit', 'Juice'], ['Orange', 'Fruit', 'Juice', 'Eat'], ['Banana', 'Fruit', 'Juice'],
 ['Banana', 'Fruit', 'Juice', 'Eat'], ['Mango', 'Fruit', 'Juice'], ['Mango', 'Fruit', 'Juice', 'Eat']]

 #using conditional expressions to assign variables
 x = 'a' if True else 'b'
 print(x)

 'a'

#First condition is evaluated, then exactly one of either a or b is evaluated and returned based
# on the Boolean value of condition. If condition evaluates to True, then a is evaluated and returned
# but b is ignored, or else when b is evaluated and returned but a is ignored.

# I want to write a single expression that returns these two dictionaries, merged. The update() method would
# be what I need, if it returned its result instead of modifying a dict in-place.
>>> x = {'a': 1, 'b': 2}
>>> y = {'b': 10, 'c': 11}
>>> z = x.update(y)
>>> print(z)
None
>>> x
{'a': 1, 'b': 10, 'c': 11}

##how to properly merge onto Z
z = {**x, **y}

##can merge wiith literal notation as well
z = {**x, 'foo': 1, 'bar': 2, **y}

#What is the most elegant way to check if the directory a file is going to be written to exists,
# and, if not, create the directory using Python?
from pathlib import Path
Path("/my/directory").mkdir(parents=True, exist_ok=True)
#pathlib.Path.mkdir as used above recursively creates the directory and does not raise an exception if
#the directory already exists. If you don't need or want the parents to be created, skip the parents argument

##Python doesn't have a str.contains method, so you can use 'in' statement ro find != -1
if "blah" not in somestring: 
    continue
#OR
s = "This be a string"
if s.find("is") == -1:
    print "No 'is' here!"
else:
    print "Found 'is' in the string."

#how to check if a list is empty
if not a:
  print("List is empty")

##how to sort items in a dict based on value, not by keys
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
{k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
{0: 0, 2: 1, 1: 2, 4: 3, 3: 4}

##Differene between .append and extend in a list
x = [1, 2, 3]
x.append([4, 5])
print (x)
[1, 2, 3, [4, 5]]
#extend appends elements from an iterable
x = [1, 2, 3]
x.extend([4, 5])
print (x)
[1, 2, 3, 4, 5]

##how to make a flat list out of a list of lists (given nested list 'l')
flat_list = [item for sublist in l for item in sublist]
#which means:
flat_list = []
for sublist in l:
    for item in sublist:
        flat_list.append(item)

###simple adding two lists together
listone = [1,2,3]
listtwo = [4,5,6]

joinedlist = listone + listtwo


###FULL process of finding all .csv(in this case .ipynb) and returning them as snakecase strings to be read in
import os
l = os.getcwd()
k = os.listdir(l)

for i in k:
    check = re.search('([\w+\s?]+)(.ipynb)', i, flags = re.IGNORECASE)
    if check:
        print(check.group(1).strip().replace(' ','_').lower()) ##instead of print, read_csv()
RESULT:
comment_your_code
fb_coding_challenge
leetcode_challenges

##how to read a file line-by-line into a list
with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

###More efficient way of reading lines into a list
lines = [line.rstrip('\n') for line in open('test_file.txt')]
RETURNS: 
['Here we go with the random notes',
 'to check if I am doing the ',
 'operations correctly',
 '',
 "Let's see how this goes"]

 ##How to delete a file or directory straight from python
os.remove() - removes a file.
os.rmdir() - removes an empty directory.
shutil.rmtree() - deletes a directory and all its contents.
#Path objects from the Python 3.4+ pathlib module also expose these instance methods:
pathlib.Path.unlink() - removes a file or symbolic link.
pathlib.Path.rmdir() - removes an empty directory.

