import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='poster', 
        style='whitegrid',)


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
    
def draw_allnumbers():
    '''Draw the numbers for the whole game at once'''
    return np.random.choice(range(1, 76), size=75, replace=False)

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


if __name__ == '__main__':
    # Let's play several rounds
    n_players = 100 
    n_rounds = 100
    games = [play_game(n_players=n_players) for _ in range(n_rounds)]

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (lines, bingos) in enumerate(games):
        ax.plot(lines/n_players, color='steelblue', linewidth=1, alpha=0.1,)
        ax.plot(bingos/n_players, color='orange', linewidth=1, alpha=0.1,)
    lines, bingos = zip(*games)
    lines, bingos = np.array(lines), np.array(bingos)
    sns.tsplot(lines/n_players, color='steelblue', ax=ax, condition='line',
               linewidth=4,)
    sns.tsplot(bingos/n_players, color='orange', ax=ax, condition='bingo',
               linewidth=4,)
    plt.legend()   
    plt.ylabel(f'proportion of players with line/bingo')
    plt.xlabel('number of numbers drawn')
    plt.grid(linestyle='--', alpha=0.8)
    plt.tight_layout()
    sns.despine()
    plt.savefig('images/lines_and_bingos.png')
    plt.close()