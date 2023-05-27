import pdb
import numpy as np
from sklearn.model_selection import ParameterGrid
from env.driver import driver_env_arr

env = driver_env_arr(n_drivers=100)


def calc_price (month_lambda, base_price=15, m_factor=1, l_factor=1):
    month = month_lambda[0]
    lam = month_lambda[1]
    price = base_price-(month*m_factor)+(lam*l_factor)
    price = max(price,0)
    price = min(price,30)
    return price


# Define the parameter grid
param_grid = {'m_factor': [1e-1, 1e0], 
              'l_factor': [1e-1, 1e0],
              'base_price': [1, 5, 15]}


def profit_model(m_factor,l_factor,base_price):
    print('m_factor: {:.2f}\tl_factor: {:.2f}\tbase_price: {:.2f}'.format(m_factor,l_factor,base_price))
    total_profit = (-1)*30*1000
    state = env.reset()
    total_profit += (30-base_price)*state.shape[0]

    for m in range(1,12):
        # pdb.set_trace()
        actions = np.apply_along_axis(calc_price,1,state, base_price=base_price, m_factor=m_factor, l_factor=l_factor)
        total_profit += env.step(actions)
        state, added = env.add_riders(m)
        if added:
            total_profit -= 1000*30
        print(f'Total Profit: {total_profit:.2f}\tMonth: {m}')

    return total_profit
best_profit = 0
best_params = {}

for g in ParameterGrid(param_grid):
  _profit = profit_model(**g)
  if best_profit < _profit:
    best_profit = _profit
    best_params = g
    print('\nBest Profit: {:.2f}\n'.format(best_profit))