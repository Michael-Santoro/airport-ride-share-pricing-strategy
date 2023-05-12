

# Airport Ride Service Problem
**Michael Santoro - michael.santoro@du.edu**

Worked solution to the following.

## Problem Statement
You're launching a ride-hailing service that matches riders with drivers for trips between the Toledo Airport and Downtown Toledo. It'll be active for only 12 months. You've been forced to charge riders \$30 for each ride. You can pay drivers what you choose for each individual ride.
# New Section
The supply pool (“drivers”) is very deep. When a ride is requested, a very large pool of drivers see a notification informing them of the request. They can choose whether or not to accept it. Based on a similar ride-hailing service in the same market, you have some [data](https://docs.google.com/spreadsheets/d/1gEMVOCXvWBcoUsTnfgDazKdur09KvnUDKW_9BE_XOD0/edit#gid=2115831440) on which ride requests were accepted and which were not. (The PAY column is what drivers were offered and the ACCEPTED column reflects whether any driver accepted the ride request.)

The demand pool (“riders”) can be acquired at a cost of $30 per rider at any time during the 12 months. There are 10,000 riders in Toledo, but you can't acquire more than 1,000 in a given month. You start with 0 riders. “Acquisition” means that the rider has downloaded the app and may request rides. Requested rides may or may not be accepted by a driver. In the first month that riders are active, they request rides based on a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) where lambda = 1. For each subsequent month, riders request rides based on a Poisson distribution where lambda is the number of rides that they found a match for in the previous month. (As an example, a rider that requests 3 rides in month 1 and finds 2 matches has a lambda of 2 going into month 2.) If a rider finds no matches in a month (which may happen either because they request no rides in the first place based on the Poisson distribution or because they request rides and find no matches), they leave the service and never return.

Submit a written document that proposes a pricing strategy to maximize the profit of the business over the 12 months. You should expect that this singular document will serve as a proposal for
1. A quantitative executive team that wants to know how you're thinking about the problem and what assumptions you're making but that does not know probability theory
1. Your data science peers so they can push on your thinking
Please submit any work you do, code or math, with your solution.
