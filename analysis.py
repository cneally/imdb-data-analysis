'''
Author: @cneally
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd


movies = pd.read_csv('movie_metadata.csv')
movies['movie_title'] = movies['movie_title'].str.replace(u'\xa0','')
movies.fillna(0)

def main_menu():
    
    print("Please choose one of the following options: \n")
    print("1. Most successful directors or actors")
    print("2. Film comparison")
    print("3. Analyse the distribution of gross earings")
    print("4. Genre Analysis")
    print("5. Earnings and IMDD scores")
    print("6. Exit")
    choice = int(input(">"))
    
    return choice
    
def main_menu_selection(choice):
    
    if choice == 1:
        option = question1_sub_menu()
        question1_menu_validation(option)
    elif choice == 2:
        option = question2_sub_menu()
        question2_menu_validation(option)
    elif choice == 3:
        dates = get_dates()
        distribution_of_gross_earnings(dates[0], dates[1])
    elif choice == 4:
        get_mean_score_by_genre()
    elif choice == 5:
        imdb_gross_prediction()
        imdb_facebook_prediction()
        imdb_director_likes_prediction()
    elif choice == 6:
        print("Thank you for using the data analysis application")
        exit()
       
        

def question1_sub_menu():
    
    print("1. Top Directors")
    print("2. Top Actors")
    choice = int(input(">"))
    
    return choice


def question1_menu_validation(n):
    
     if n == 1:
            x = get_n()
            get_directors_question1(x)
     elif n == 2:
            x = get_n()
            get_actors_question1(x) 
     else:
            print('That is not a valid option')
    
    
def question2_sub_menu():  
    
    print("1. IMDB Scores")
    print("2. Gross Earning")
    print("3. Movie Facebook Like")
    choice = int(input(">"))
    
    return choice


def question2_menu_validation(n):
    
     films = get_movies_for_comparison()
     if n == 1:
            imdb_score_question2(films[0], films[1])
     elif n == 2:
             gross_earnings_question2(films[0], films[1])
     elif n == 3:
             facebook_likes_question2(films[0], films[1])
     else:
            print('That is not a valid option')
    
    

def get_movies_for_comparison():
    film1 = input('Please enter the first movie > ')
    film2 = input('Please enter the second movie > ')
    
    return film1, film2


def get_dates():
    
    start_date = int(input('Please enter the start year > ' ))
    if start_date > 2016 or start_date < 1920:
        print('error')
        start_date = int(input('Please enter the start year > ' ))
    
    end_date = int(input('Please enter the end year > ' ))
    if end_date > 2016 or end_date < 1920:
        print('error')
        end_date = int(input('Please enter the end year > ' ))

    return start_date, end_date


def get_n():
    
    n = int(input('Please enter the number of results you want to return '))
    
    while n < 1:
        print('Error')
        n = int(input('Please enter the number of results you want to return '))
    if n > 1:
        return n


def get_directors_question1(n):
    
      directors = movies[['director_name', 'gross']]
      directors = directors.dropna()
      directors.columns = ['Directors', 'Gross Earning']
      director_grouping = directors.groupby('Directors').sum().sort_values('Gross Earning', ascending = False )
      top_n_directors = pd.DataFrame(director_grouping[:n])
      top_n_directors.columns = ['Gross Earnings']
      
      print('The Top', n, ' Highest Earning Directors are:')
      print(top_n_directors)
      top_n_directors.plot.bar() 
      plt.show()
      
      
def get_actors_question1(n): 
    
      actors = movies[['actor_1_name', 'gross']]
      actors = actors.dropna()
      actors.columns = ['Actors', 'Gross Earning']
      actor_grouping = actors.groupby('Actors').sum().sort_values('Gross Earning', ascending = False )
      top_n_actors = pd.DataFrame(actor_grouping[:n])
      top_n_actors.columns = ['Gross Earnings']
      
      print('The Top', n, ' Highest Earning Actor are:')
      print(top_n_actors)
      top_n_actors.plot.bar() 
      plt.show()
    

def imdb_score_question2(str1, str2):
    
    # Creating a new DataFraee from likes and movies where the movie matched the movies the user has entered
    score1 = movies[['movie_title', 'imdb_score']][movies['movie_title']==str1]
    score2 = movies[['movie_title', 'imdb_score']][movies['movie_title']==str2]
    
    #concatenating the results so the they can ebe plotted on the bar graph
    comparison = [score1, score2]
    result = pd.concat(comparison)
    result.columns = ['Movie', 'IMDb Rating']
    result.plot.bar(x='Movie', y='IMDb Rating')
    plt.show()
    print(result)


def gross_earnings_question2(str1, str2):
    
    # Creating a new DataFrame from likes and movies where the movie matched the movies the user has entered
    score1 = movies[['movie_title', 'gross']][movies['movie_title']==str1]
    score2 = movies[['movie_title', 'gross']][movies['movie_title']==str2]
    
    #concatenating the results so the they can ebe plotted on the bar graph
    comparison = [score1, score2]
    result = pd.concat(comparison)
    result.columns = ['Movie', 'Gross Earnings']
    result.plot.bar(x='Movie', y='Gross Earnings')
    plt.show()
    print(result)



def facebook_likes_question2(str1, str2): 
    
    # Creating a new DataFraee from likes and movies where the movie matched the movies the user has entered
    score1 = movies[['movie_title', 'movie_facebook_likes']][movies['movie_title']==str1]
    score2 = movies[['movie_title', 'movie_facebook_likes']][movies['movie_title']==str2]
    
    #concatenating the results so the they can ebe plotted on the bar graph
    comparison = [score1, score2]
    result = pd.concat(comparison)
    result.columns = ['Movie', 'Facebook Likes']
    result.plot.bar(x='Movie', y='Facebook Likes')
    plt.show()
    
    print(result)


def distribution_of_gross_earnings(start, finish): 
    
    
    ge = movies[['title_year', 'gross']]
    ge.fillna(0)
    # gettig the max, min, mean of the ge DF and sorting the values by title_year and 
    # gross
    ge_group_max = ge.groupby('title_year').max().sort_values(['title_year','gross'], ascending = [False, False] )
    ge_group_min = ge.groupby('title_year').min().sort_values(['title_year','gross'], ascending = [False, False] )
    ge_group_mean = ge.groupby('title_year').mean().sort_values(['title_year','gross'], ascending = [False, False] )
    
    # The below is returning the the column values between the start and finish year entered in by the users
    # and print those results to the screen and plot them in the graph
    print('\nMaximum\n', ge_group_max.loc[start:finish], end='\n')
    print('\nMinimum\n',ge_group_min.loc[start:finish], end='\n')
    print('\nMean\n',ge_group_mean.loc[start:finish], end='\n')
    
    # The following block of code is plotting the results and displaying it in a line graph
    plt.plot(ge_group_max.loc[start:finish], marker='o',color='g',linestyle='-',label='max')
    plt.plot(ge_group_mean.loc[start:finish],marker='o',color='r',linestyle='-',label='mean')
    plt.plot(ge_group_min.loc[start:finish],marker='o',color='b',linestyle='-',label='min')
    plt.xlabel('title_year', fontsize=18)
    plt.ylabel('Gross Earnings', fontsize=18)
    plt.legend(loc = 'upper left')
    plt.show()
    
    

def get_genre(): 

    # listing all the genres to display the user
    genre =  movies['genres'].str.split('|')
    set = genre.apply(pd.Series).stack().value_counts()
    gdict = dict(set)
    l = list(gdict)
    
    for l in l:
        print(l)
    
    return input('Please a choose one of the above genres > ' )
  


def get_mean_score_by_genre():

    genredf = movies[['genres', 'imdb_score']]
    
    user_genre = get_genre()
    # Splitting the genres up by using '\' delimiter
    genre =  movies['genres'].str.split('|')
    # the following line stacks the the genre on top of one another and counts the occurrence of eache genre
    genre = genre.apply(pd.Series).stack().value_counts()
    #orgainising the data by genre and calculatiing the mean imdb score using genres
    mean = genredf.groupby('genres').mean().sort_values(['genres','imdb_score'], ascending = [False, False] )
    
    mean.plot.line()
    plt.xlabel(user_genre, fontsize=18)
    plt.show()
    
    print(mean.loc[user_genre])


def imdb_gross_prediction():
   
    gross_votes = pd.DataFrame(movies[['gross','num_voted_users']])
    gross_score  = pd.DataFrame(movies[['imdb_score','gross']] )
    
   # Dops NAN values from frames in the data frame
    gross_votes = gross_votes.dropna()
    gross_score = gross_score.dropna()

    # Adding the required data to numpy arrays for testing 
    x = np.array([gross_votes.gross])
    y = np.array([gross_score.imdb_score])

    # Shape the data so it is the same size so that it can suitable for linear regression
    x=np.array(x).reshape((-1,1))
    y=np.array(y).reshape((-1,1))
 
    # Creating linear model to test and train data to make predictions using Linear Regressions
    linmodel = linear_model.LinearRegression()

    #split the data in to test data and training data  and specify size of data whihch  epresents the absolute number of test sample
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
     
    # Here the data is passed to through to the model r=to be trained
    linmodel.fit(x_train,y_train)
    
    #The prediction made based on the data passed through
    prediciton = linmodel.predict(x_test)

    # Plotting the data for Visualisation
    fig,axis = plt.subplots()
   
    axis.scatter(y_test,prediciton,color = 'blue')
    axis.plot(y_test,y_test ,color = 'green')
    axis.set_title("Score predcition based on gross earnings",color = 'red')
    axis.set_xlabel("Predicted Score",color = 'red')
    axis.set_ylabel("Gross Earnings",color = 'red')
    plt.show()


def imdb_facebook_prediction():
    
    gross_likes = pd.DataFrame(movies[['movie_facebook_likes','num_voted_users']])
    gross_score  = pd.DataFrame(movies[['imdb_score','movie_facebook_likes']] )
    
   # Dops NAN values from frames in the data frame
    gross_likes = gross_likes.dropna()
    gross_score = gross_score.dropna()

    # Adding the required data to numpy arrays for testing 
    x = np.array([gross_likes.movie_facebook_likes])
    y = np.array([gross_score.imdb_score])

    # Shape the data so it is the same size so that it can suitable for linear regression
    x=np.array(x).reshape((-1,1))
    y=np.array(y).reshape((-1,1))
 
    # Creating linear model to test and train data to make predictions using Linear Regressions
    linmodel = linear_model.LinearRegression()

    #split the data in to test data and training data  and specify size of data whihch  epresents the absolute number of test sample
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
     
    # Here the data is passed to through to the model r=to be trained
    linmodel.fit(x_train,y_train)
    
    #The prediction made based on the data passed through
    prediciton = linmodel.predict(x_test)

    # Plotting the data for Visualisation
    fig,axis = plt.subplots()
   
    axis.scatter(y_test,prediciton,color = 'blue')
    axis.plot(y_test,y_test ,color = 'green')
    axis.set_title("Score predcition based on Movie Facebook Likes",color = 'red')
    axis.set_xlabel("Predicted Score",color = 'red')
    axis.set_ylabel("Facebook Likes",color = 'red')
    plt.show()

def imdb_director_likes_prediction():
    
    gross_budget = pd.DataFrame(movies[['director_facebook_likes','num_voted_users']])
    gross_score  = pd.DataFrame(movies[['imdb_score','director_facebook_likes']] )
    
   # Dops NAN values from frames in the data frame
    gross_budget = gross_budget.dropna()
    gross_score = gross_score.dropna()

    # Adding the required data to numpy_ arrays for testing 
    x = np.array([gross_budget.director_facebook_likes])
    y = np.array([gross_score.imdb_score])

    # Shape the data so it is the same size so that it can suitable for linear regression
    x=np.array(x).reshape((-1,1))
    y=np.array(y).reshape((-1,1))
 
    # Creating linear model to test and train data to make predictions using Linear Regressions
    linmodel = linear_model.LinearRegression()

    #split the data in to test data and training data  and specify size of data whihch  epresents the absolute number of test sample
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
     
    # Here the data is passed to through to the model r=to be trained
    linmodel.fit(x_train,y_train)
    
    #The prediction made based on the data passed through
    prediciton = linmodel.predict(x_test)

    # Plotting the data for Visualisation
    fig,axis = plt.subplots()
   
    axis.scatter(y_test,prediciton,color = 'blue')
    axis.plot(y_test,y_test ,color = 'green')
    axis.set_title("Score predcition based on Director Facebook Likes",color = 'red')
    axis.set_xlabel("Predicted Score",color = 'red')
    axis.set_ylabel("Budget",color = 'red')
    plt.show()


def main():
    
    choice = 0
    while choice != 6:
        choice = main_menu()
        while choice not in range(0, 7):
            print('ERROR: Invalid selection please choose between [1-6] > ')
            choice = main_menu()
        main_menu_selection(choice)
        if choice == 6:
            print('Thank You, Good Bye')
            exit()


main()