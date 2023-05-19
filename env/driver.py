import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pdb


class driver_env():
    '''
    This is the enviroment class for modeling of the rideshare problem.
    
    Observation: Consists of vector of two parameters lambda, month.
    
    Action: Consists of the offered price to the driver.

    train_mode: Train mode set to 'True' will complete the entire year at that chosen price. 
    Setting train mode to 'False' will complete a single month and return profit for that month.
    
    '''
    def __init__(self, n_drivers=1000, train_mode=True):
        self.n_drivers = n_drivers
        self.train_mode = train_mode
        self.fit_model()


    def sample_poisson(self, lambd, size=1):
        """
        Generate random samples from a Poisson distribution with parameter lambda.

        Parameters:
            lambd (float): The parameter lambda of the Poisson distribution.
            size (int, optional): The number of samples to generate (default 1).

        Returns:
            np.ndarray: An array of random samples from the Poisson distribution.
        """
        return np.random.poisson(lam=lambd, size=size)[0]
    
    def fit_model(self):

        df = pd.read_csv('env/driverAcceptanceData - driverAcceptanceData.csv',index_col=0)


        # Split data into features and target
        X = df.drop('ACCEPTED', axis=1).values
        y = df['ACCEPTED'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit logistic regression model
        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = lr.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        self.lr = lr

    def driver_desc(self, price):
        return np.any(np.random.choice([False, True], size=self.n_drivers, p=self.lr.predict_proba(np.array([price]).reshape(-1, 1))[0]))
    

    def reset(self, month=None, lam=1, train_mode=True):
        self.train_mode = train_mode
        if month == None:
            self.month = random.sample(range(12), 1)[0]
        self.lam = self.sample_poisson(lam)
        return [self.month,self.lam]
    
    def calc_reward_buffer(self,buffer):
        pdb.set_trace()
        old_reward = [i[0] for i in buffer]

        new_reward = [sum(old_reward[i:]) for i in range(len(old_reward))]

        for j in range(len(buffer)):
            buffer[j][0] = new_reward[j]
        pdb.set_trace()
        return buffer

    def step(self,action):
        buffer = []
        ## Return: [reward, state, done]
        while True:
            ## Increase Month
            self.month += 1

            ## Calc Profit
            reward = self.lam*(30-action[0])

            ## Get Drivers Choice
            driver_accepted = self.driver_desc(action[0])

            pdb.set_trace()
            if not driver_accepted:
                buffer.append([0,np.array([self.month,0]),True])
                return self.calc_reward_buffer(buffer)

            if self.month == 12:
                buffer.append([reward,np.array([self.month,0]),True])
                return self.calc_reward_buffer(buffer)
    
            ## Get New Lambda
            self.lam = self.sample_poisson(self.lam)
            if self.lam == 0:
                buffer.append([reward,np.array([self.month,self.lam]),True])
                if buffer:
                    return self.calc_reward_buffer(buffer)
                return [[reward,np.array([self.month,self.lam]),True]]
            pdb.set_trace()
            buffer.append([reward,np.array([self.month,self.lam]),False])

class driver_env_arr():
    '''
    This is the enviroment class for modeling of the rideshare problem.
    
    Observation: Consists of vector of two parameters lambda, month.
    
    Action: Consists of the offered price to the driver.

    train_mode: Train mode set to 'True' will complete the entire year at that chosen price. 
    Setting train mode to 'False' will complete a single month and return profit for that month.
    
    '''
    def __init__(self, n_drivers=1000, train_mode=True, max_riders=1e4, activated_riders=1e3):
        self.n_drivers = n_drivers
        self.train_mode = train_mode
        self.max_riders = max_riders
        self.riders_activated = activated_riders
        self.total_riders = 0
        self.fit_model()
    
    def fit_model(self):

        df = pd.read_csv('env/driverAcceptanceData - driverAcceptanceData.csv',index_col=0)


        # Split data into features and target
        X = df.drop('ACCEPTED', axis=1).values
        y = df['ACCEPTED'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit logistic regression model
        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = lr.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        self.lr = lr

    def driver_desc(self, price):
        def choice(p):
            return np.random.choice([False, True], size=self.n_drivers, p=p)
        price = price.reshape(price.shape[1],1)
        choices = np.apply_along_axis(choice, 1, self.lr.predict_proba(price))
        return np.any(choices,axis=1)
    

    def reset(self, month=0, lam=1, riders=1000):
        self.total_riders += riders
        month = np.zeros((riders,1))
        lam = np.ones((riders,1))
        lam = np.random.poisson(lam)
        self.states = np.hstack((month, lam))
        self.states = self.states[self.states[:,1] != 0]
        return self.states

    def step(self,action):
                ## Return: [reward, state, done]

        ## Increase Month
        self.states[:,0] += 1
     
        ## Calc Profit
        reward = 30-action
        reward = reward.reshape(reward.shape[1],)

        ## Get Drivers Choice
        driver_accepted = self.driver_desc(action)
        reward = reward[driver_accepted]
        self.states = self.states[driver_accepted]
  
        ## Get New Lambda
        self.states[:,1] = np.random.poisson(self.states[:,1])
        ## Drops any states with lam = 0
        self.states = self.states[self.states[:,1] != 0]
        return np.sum(reward)
    
    def add_riders(self,month):
        if self.total_riders >= self.max_riders:
            return self.states
        self.total_riders += self.riders_activated
        m = np.ones((int(self.riders_activated),1))
        m = month*m
        lam = np.ones((int(self.riders_activated),1))
        lam = np.random.poisson(lam)
        states = np.hstack((m, lam))
        states = states[states[:,1] != 0]
        self.states = np.vstack((self.states, states))
        return self.states









