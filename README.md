# Let's play bingo percolation! 

It's not coffee, but still percolates. It's Bingo!

_Disclaimer:_ what you're about to read might be totally trivial and analytically
solvable after a tiny bit of combinatorics/probabilities gymnastics. 
If that already sounds boring, imagine actually going through it with pen and 
paper. Let's see what Python can do for us.

_Disclaimer 2:_ this is not about coffee. I'm sorry too.


## The story
So one day we where playing champagne-bingo because.. why not?
It's a pretty straight forward game. 
If you know [bingo](https://en.wikipedia.org/wiki/Bingo_(U.S.)), it's the same,
 but! everyone has a glass of champagne and is supposed to take a sip if the 
 drawn number is not on his/her ticket. Easy.
You can do the math for yourself about the number of times you end up having
to drink..but that's not the point here.

### Percolation
[Percolation](https://en.wikipedia.org/wiki/Percolation_theory) refers to
 a whole world of interesting stuff, ranging from coffee brewing to epidemics 
 spreading and everything in between. 
But it suffices here to know that the deal is about trying to understand 
_phase transitions_ in a particular system.
That means, characterizing and being able to predict a qualitative change in 
the behaviour of some macroscopic property of the system. 
For example, let's say we have a bunch of isolated new users of a new social 
network.
Now we start connecting them _randomly_, like getting them to be friends but 
without any criterion rather than a certain fixed probability of getting connected.
Percolation theory helps us predicting what is the minimum amount of friends per
 person that we need (on average) in order to have a social network that is 
 connected, i.e., in which everybody is reachable by taking a finite number of
 hops in the network.
Pretty useful, right? 

### Back to the game 
It turns out we _did not stop playing_ when we where
supposed to, i.e., whenever someone won (got bingo) just for the sake of.. 
you know.. just having fun a bit longer.

And my observation was that, after a long period of people doing little more 
than drinking number after number (pretty hard life), a rather abrupt transition 
seemed to occur and folks were getting (first) lines (and then) bingo more and 
more often.
In other words, as time passed (numbers got drawn), the higher was the
 probability that someone would jump up, scream out loud "line" or "bingo" 
 while performing a nice [hula dance](https://www.youtube.com/watch?v=r3JAM1nuNAk).
Yes, you got it, a non-linear transition. 
Yes, like a percolation. To be more precise, a bingo-percolation.

So let's see if that is actually the case.


Bingo 
-----
There are a couple of variations to it, but the one we played looked like this:
- 75 balls numbered 1 to 75
- each ticket has 25 numbers (5x5 grid), each column having the numbers:
    - (B) first:  1-15
    - (I) second: 16-30
    - (N) third:  31-45
    - (G) fourth: 46-60
    - (O) fifth:  61-75 
- "line": full horizontal/vertical/diagonal line marked *
- "bingo": full ticket marked

(_Minutia:_ For the "lines" we only track the occurrence of _at least_ one line per ticket,
since there was a price for only the first one getting it.
So after a player gets a line, she will afterwards always count as simply having
at least one line, independently from the fact that she can get a different line 
in the future.)

### On to the code!
#### Requirements
- `numpy`
- `matplotlib`
- `seaborn`

```python
# We need
# - tickets
# - number drawer
# - ticket marker
# - ticket checker 
# - game runner


def make_tickets(n_tickets=10):
    '''
    Return 3D array - stack of n_tickets.
    ''' 
    tickets = [
        np.transpose([
            sorted(np.random.choice(row, size=5, replace=False))
            for row in np.arange(1, 76).reshape(5, 15)])
        for _ in range(n_tickets)
        ]
    if n_tickets == 1:
        return np.array(tickets)[0]
    return np.array(tickets) 

def draw_allnumbers():
    '''Draw the numbers for the whole game at once'''
    return np.random.choice(range(1, 76), size=75, replace=False)

def mark_ticket(ticket, number):
    '''
    Modify ticket putting a mark (0) if number is on it.
    '''
    ticket[ticket == number] = 0
    return ticket

def check_line(ticket):
    '''
    Return: (bool) ticket has at least one of the valid lines marked.
    '''
    line = any([
        sum([np.all(row == 0) for row in ticket]) > 0,  # horizontal
        sum([np.all(row == 0) for row in ticket.T]) > 0,  # vertical
        np.all(np.diag(ticket) == 0),  # diagonals
        np.all(np.diag(np.fliplr(ticket)) == 0)
    ])
    return line

def check_bingo(ticket):
    '''Return True if ticket has bingo (all numbers are marked)'''
    return np.all(ticket == 0) 
    
def play_game(n_players=100):
    # Initialize game
    tickets = make_tickets(n_players)
    numbers = draw_allnumbers()

    # Track number of lines/bingos after each number drawn
    lines = np.zeros_like(numbers)
    bingos = np.zeros_like(numbers)

    # Play rounds, number by number
    for i, number in enumerate(numbers):
        for ticket in tickets:
            mark_ticket(ticket, number)
            # Check how many players got have at least one line or bingo
            lines[i] += check_line(ticket)
            bingos[i] += check_bingo(ticket)
    
    return lines, bingos
```

### That's it!
So we can run the script bingo.py
```
python bingo.py
```

The output figure shows the results of 100 rounds of the game, played by
100 players.
Individual traces correspond to each run, while the bold lines depict the 
average trace for each case.

![png](https://github.com/fabridamicelli/bingo_percolation/blob/master/images/lines_and_bingos.png)

### Lo and behold, the curves kind of agree with the initial intuition :)

