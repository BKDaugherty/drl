# Vanilla DQN in Rust

This is an implementation of a Vanilla DQN as described [here](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), but in Rust!

## Why Rust?

I don't really like writing Python code, and find the developer time to take too long, so I decided I'm going to fuel my ML journey with a type-safe langauge. Maybe I'll write bindings if someone makes me write Python, but I'd really just prefer not to :P

## Why this algorithm?

We read it in my paper club, and I wanted to implement it!

## Installation

This depends on [tch-rs](https://github.com/LaurentMazare/tch-rs), so follow instructions there first.
It also depends on tensorflow. [Try following these instructions](https://github.com/tensorflow/rust).

Specifically you'll want to run `cargo build -j 1` once you've done the above.


## TODO
- Initialize algorithm
- Need to see how to train a Pytorch algo in general
- Check this tutorial out: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
