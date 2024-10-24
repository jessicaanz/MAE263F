# MAE263F Homework 1

## Assignment 1
Download Problem1.py and open the code in an editor of choice.

1. Navigate to the section of code labeled **MAIN**. There is a boolean variable called `implicit` which controls if an implict or explicit simulation is run. Set the boolean equal to 1 to run the implicit simulation, and set it to 0 to run the explicit simulation. There are 2 more booleans called `plot_all` and `plot_select`. Set `plot_all` equal to 1 to plot every 0.5 seconds, and set `plot_select` equal to 1 to plot the times requested in the assignment. Run the script to run the simulation.
2. After running the script, the terminal velocity will be printed in the terminal.
3. In the **MAIN** section of the code, there are variables `R1`, `R2` and`R3`. Set these variables to the same value and run the simulation to test what happens when all nodes have equal radii.
4. In the **MAIN** section of the code, there is an if-else statement labeled "Time Step". In the section labeled "explicit" change the value of `dt` and run the simulation to test various time steps.

## Assignment 2
Download Problem2.py and open the code in an editor of choice.

1. Navigate to the section of code labeled **MAIN** where there are booleans called `plot_all` and `plot_select`. Set `plot_all` equal to 1 to plot every 0.5 seconds, and set `plot_select` equal to 1 to plot the figures requested in the assignment. Run the script to run the simulation. The terminal velocity will be printed in the terminal.
2. The final deformed shape will be plotted when the simulation is run with `plot_select` equal to 1.
3. Values for the terminal velocity with respect to various numbers of nodes and time step sizes have been recorded in the code. Run the simulation to view plots of this data.

## Assignment 3
Download Problem3.py and open the code in an editor of choice.

1. Run the code to view plots of the deformed beam shape and maximum vertical displacement vs. time. Both the simulated and theoretical y_max will be printed in the terminal upon completion of the script.
2. Navigate to the section of code labeled **MAIN** and find the section labeled "load". Change the value of the variable `load` to 20,000 and run the simulation to see how a larger load changes the results.